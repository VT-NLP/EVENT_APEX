import json
import logging
import os
# logger
import sys
import time
from datetime import datetime

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import TriggerDetection
from utils import Metadata, calculate_f1, pred_to_event_mention, pred_to_event_mention_novel, \
    pack_data_to_trigger_model_joint, load_from_jsonl
from utils.config import Config
from utils.optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename=os.getenv('LOGFILE'))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main(**kwargs):
    # configuration
    config = Config()
    config.update(**kwargs)
    logging.info(config)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(39)
    if config.ace:
        config.metadata = Metadata().ace
    elif config.ere:
        config.metadata = Metadata().ere
    elif config.maven:
        config.metadata = Metadata().maven
    else:
        raise NotImplemented
    # load data
    te_dataset = torch.load(config.te_dataset)
    dev_dataset = torch.load(config.dev_dataset)
    tr_dataset = torch.load(config.tr_dataset)
    te_json = load_from_jsonl(config.te_json)
    dev_json = load_from_jsonl(config.dev_json)
    # model setup
    model_trigger = TriggerDetection(config)
    model_trigger.model.resize_token_embeddings(len(config.tokenizer))
    if config.use_gpu:
        model_trigger.cuda()

    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.eval_batch_size)
    te_loader = DataLoader(te_dataset, shuffle=False, batch_size=config.eval_batch_size)

    if config.load_pretrain and not config.do_train:
        model_trigger.load_state_dict(torch.load(config.pretrained_model_path))
        eval_trigger(model_trigger, te_loader, config, te_json)
        return 0

    # optimizer
    param_optimizer1 = list(model_trigger.model.named_parameters())
    param_optimizer1 = [n for n in param_optimizer1 if 'pooler' not in n[0]]
    param_optimizer2 = list(model_trigger.linear.named_parameters())
    param_optimizer2.append(('W', model_trigger.W))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay, 'lr': config.lr * 3},
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    N_train = torch.sum(tr_dataset.tensors[-1] < config.metadata.event_count) + config.sampling * len(
        tr_dataset)
    num_train_steps = int(N_train / config.train_batch_size / config.gradient_accumulation_steps * config.EPOCH)
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.lr,
                         warmup=config.warmup_proportion,
                         t_total=t_total)

    # loss
    weights = torch.ones(2).cuda()
    weights[0] = config.non_weight
    weights[1] = config.trigger_training_weight
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1, reduction='mean')

    f1, pre_f1 = 0, 0
    global_step = [0]

    best_model = ['']
    for epoch in range(config.EPOCH):
        logging.info('==============')
        logging.info('Training at ' + str(epoch) + ' epoch')

        # sample different negative examples each epoch
        pos_tokens = torch.tensor([torch.any(x[-1] < config.metadata.event_count) for x in tr_dataset])
        random_neg_tokens = torch.rand(pos_tokens.shape) < config.sampling
        _tr_dataset = TensorDataset(*tr_dataset[pos_tokens + random_neg_tokens])
        tr_loader = DataLoader(_tr_dataset, shuffle=True, batch_size=int(config.train_batch_size))

        f1 = train_trigger(config, model_trigger, epoch, pre_f1,
                           tr_loader, criterion, optimizer, t_total, global_step,
                           dev_loader, dev_json, best_model)
        if f1 > pre_f1:
            pre_f1 = f1

    if best_model[0]:
        # evaluate on test set
        model_trigger.load_state_dict(torch.load(best_model[0]))
        f1, precision, recall, date_time = eval_trigger(model_trigger, te_loader, config, te_json)
        logging.info('Test results')
        logging.info('time: {}'.format(date_time))
        logging.info('f1_bio: {} |  p:{}  | r:{}'.format(f1, precision, recall))

    return 0


def train_trigger(config, model, epoch, pre_f1, tr_loader, criterion, optimizer, t_total, global_step,
                  eval_loader=None, eval_json=None, best_model=None):
    logging.info("Begin trigger training...")
    logging.info("Bert model: {}\nBatch size: {}".format(str(config.pretrained_weights), config.train_batch_size))
    logging.info("Epoch {}".format(epoch))
    logging.info("time: {}".format(time.asctime()))

    model.zero_grad()
    f1_new_best, model_new_best = pre_f1, None

    num_batchss = len(tr_loader)
    eval_step = int(num_batchss / config.eval_per_epoch)
    for i, batch in enumerate(tqdm(tr_loader)):
        # Extract data
        dataset_id, bert_sentence_in, triggers, idxs_to_collect_sent, idxs_to_collect_event, sent_lengths, \
        sent_ids_long, bert_sentence_lengths, pos_tags, embedding_length, \
            = pack_data_to_trigger_model_joint(batch)

        # forward
        feats = model(dataset_id, bert_sentence_in, idxs_to_collect_sent.long(), idxs_to_collect_event.long(),
                      bert_sentence_lengths, pos_tags)

        # Loss
        targets = triggers.flatten()
        targets[targets < config.metadata.event_count] = 1
        targets[targets == config.metadata.event_count] = 0
        targets[targets > config.metadata.event_count] = -1
        feats = torch.flatten(feats, end_dim=-2)

        loss = criterion(feats, targets)
        loss.backward()

        # modify learning rate with special warm up BERT uses
        if (i + 1) % config.gradient_accumulation_steps == 0:
            rate = warmup_linear(global_step[0] / t_total, config.warmup_proportion)
            for param_group in optimizer.param_groups[:-2]:
                param_group['lr'] = config.lr * rate
            for param_group in optimizer.param_groups[-2:]:
                param_group['lr'] = config.lr * 3 * rate
            optimizer.step()
            optimizer.zero_grad()
            global_step[0] += 1

        if (i + 1) % eval_step == 0:
            f1, precision, recall, date_time = eval_trigger(model, eval_loader, config, eval_json)
            if f1 > pre_f1:
                pre_f1 = f1
                f1_new_best = f1
                logging.info('New best result found for Dev.')
                logging.info('time: {}'.format(date_time))
                logging.info('epoch: {} | f1_bio: {} |  p:{}  | r:{}'.format(epoch, f1, precision, recall))
                logging.info('Save model to {}'.format(config.save_model_path + date_time))

                torch.save(model.state_dict(), config.save_model_path + date_time)
                best_model[0] = config.save_model_path + date_time

    return f1_new_best


def eval_trigger(model, dev_loader, config, gold_event):
    model.eval()
    output = []
    tp, pos, gold = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            # Extract data
            dataset_id, bert_sentence_in, triggers, idxs_to_collect_sent, idxs_to_collect_event, sent_lengths, \
            sent_ids_long, bert_sentence_lengths, pos_tags, embedding_length, \
                = pack_data_to_trigger_model_joint(batch)

            # forward
            feats = model(dataset_id, bert_sentence_in, idxs_to_collect_sent.long(), idxs_to_collect_event.long(),
                          bert_sentence_lengths, pos_tags)
            # get predictions from logits
            pred = (feats[:, :, 1] - feats[:, :, 0] - config.trigger_threshold)
            pred = [pred[k, :sent_lengths[k]] for k in range(config.eval_batch_size)]
            if config.few_shot or config.zero_shot:
                this_pred, this_pred_w_prob = pred_to_event_mention_novel(pred, config.metadata.ids_to_triggers,
                                                                          config.metadata.novel_ids)
            else:
                this_pred, this_pred_w_prob = pred_to_event_mention(pred, config.metadata.ids_to_triggers, config)
            this_pred = set(this_pred)
            this_gold = set(tuple(x) for x in gold_event[i]['event_trigger'])
            tp += len(this_gold.intersection(this_pred))
            pos += len(this_pred)
            gold += len(this_gold)
            output.append({'sentence': gold_event[i]['sentence'], "pred": list(this_pred_w_prob), "gold": this_gold})

        f1, precision, recall = calculate_f1(gold, pos, tp)
        model.train()
        now = datetime.now()
        date_time = now.strftime("%m%d%Y%H:%M:%S")

    return f1, precision, recall, date_time


if __name__ == '__main__':
    fire.Fire()
