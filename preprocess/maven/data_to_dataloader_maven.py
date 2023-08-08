# coding=utf-8
# coding=utf-8
import numpy as np
from os import listdir
from os.path import isfile, join
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from torch.autograd import Variable
import random
import re
from functools import partial
from collections import defaultdict


trigger_count = [0]


def process_maven_object(data):
    """
    read the data with pos tags
    :param data:
    :param tokenizer:
    :return:
    """
    sentence = data[0].split()
    if len(data) < 4:
        triggers = [[]]
    else:
        triggers = [x.split() for x in data[1:-2]]

    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')
    return sentence, triggers, pos, doc_id


def read_data_maven(file):
    """
    Extract data from maven BIO files
    :param file: path
    :return: a list [[sentence1, triggers, arguments], [sentence2, triggers, arguments], ...]
                triggers' length is equal to arguments length, and each arguments row corresponding to the trigger
    """
    data = open(file, 'r').read().split('\n\n')
    output_ = [i.split('\n') for i in data]
    # delete the last empty string in the data file if there is one
    if output_[-1]==['']:
        output_ = output_[:-1]

    output = list(map(process_maven_object, output_))
    return output


def pair_trigger_maven(data_bert, config):
    # get trigger type set
    data_bert_new = []
    for j in range(len(data_bert)):
        tokens, triggers, pos_tag, doc_id = data_bert[j]
        bert_sent = ['[CLS]', '[EVENT]', '[SEP]'] + config.tokenizer.tokenize(' '.join(tokens)) + ['[SEP]']
        data_tuple = (doc_id, tokens, bert_sent, triggers, pos_tag)
        data_bert_new.append(data_tuple)

    return data_bert_new


def data_extract_trigger_maven(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    sent_indx = [e[0] for e in data]
    words = [e[1] for e in data]
    bert_words = [e[2] for e in data]
    event_tags = [e[3] for e in data]
    pos_tags = [e[4] for e in data]

    return sent_indx, words, bert_words, event_tags, pos_tags


def prepare_bert_sequence(seq_batch, to_ix, pad, emb_len):
    padded_seqs = []
    for seq in seq_batch:
        pad_seq = torch.full((emb_len,), to_ix(pad), dtype=torch.int)
        # ids = [to_ix(w) for w in seq]
        ids = to_ix(seq)
        pad_seq[:len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def firstSubwordsIdx_for_one_seq(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    collected_1st_subword_idxs = []
    idx = 1  # need to skip the embedding of '[CLS]'
    for i in range(len(words)):
        w = words[i]
        w_tokenized = tokenizer.tokenize(w)
        collected_1st_subword_idxs.append(idx)
        idx += len(w_tokenized)
    collected_1st_subword_idxs.append(idx)
    collected_1st_subword_idxs.append(idx + 5)
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch(seq_batch, tokenizer):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    for seq in seq_batch:
        idx_seq = firstSubwordsIdx_for_one_seq( seq, tokenizer )
        idx_batch.append(idx_seq)
    return idx_batch


def prepare_sequence(seq_batch, to_ix, pad, seqlen, remove_bio_prefix=False):
    padded_seqs = []
    for seq in seq_batch:
        if pad == -1:
            pad_seq = torch.full((seqlen,), pad, dtype=torch.int)
        else:
            pad_seq = torch.full((seqlen,), to_ix[pad], dtype=torch.int)
        if remove_bio_prefix:
            ids = [to_ix[w[2:]] if len(w) > 1 and w[2:] in to_ix else to_ix['O'] for w in seq]
        else:
            ids = [to_ix[w] if w in to_ix else -1 for w in seq ]

        pad_seq[:len(ids)] = torch.tensor(ids, dtype=torch.int)
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def pad_trigger_pack_maven(seq_batch, to_ix, pad, seqlen):
    padded_seqs = []
    n_trigger = len(to_ix)-2
    for seq in seq_batch:
        if pad == -1:
            pad_seq = torch.full((n_trigger, seqlen), pad, dtype=torch.int)
        else:
            pad_seq = torch.full((n_trigger, seqlen), to_ix[pad], dtype=torch.int)
        if seq == [[]]:
            pass
        else:
            ids = [[to_ix[w] if w in to_ix else -1 for w in x] for x in seq]
            ids = torch.min(torch.tensor(ids), dim=0)[0]

            for i in range(len(ids)):
                if ids[i] >= to_ix['O']:
                    continue
                elif ids[i] == 0:  # 'Not' a event trigger
                    pad_seq[:, i] = 0
                    pad_seq[ids[i], i] = 1
                else:
                    pad_seq[:, i] = 0
                    pad_seq[ids[i], i] = 1

        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def data_prepare(data_bert, config):
    """
    Generate data loader for the model
    :param data_bert:
    :param tokenizer:
    :param config:
    :param word_to_ix:
    :param trigger_to_ix:
    :param split:
    :param split_ratio:
    :param shuffle:
    :return:
    """
    # data preparation
    tokenizer = config.tokenizer
    doc_id, words_batch, bert_words_batch, trigger_tags_batch, pos_tags = data_extract_trigger_maven(data_bert)
    sentence_lengths = list(map(len, words_batch))  # [len(s) for s in words_batch]
    seq_length = max(sentence_lengths)
    bert_sentence_lengths = list(map(len, bert_words_batch))  # [len(s) for s in bert_words_batch]
    bert_seq_length = max(bert_sentence_lengths)

    idxs_to_collect_batch = firstSubwordsIdx_batch(words_batch, tokenizer)

    bert_tokens = prepare_bert_sequence(bert_words_batch, config.tokenizer.convert_tokens_to_ids, config.PAD_TAG, bert_seq_length)
    trigger_tags = pad_trigger_pack_maven(trigger_tags_batch, config.metadata.triggers_to_ids, -1, seq_length)

    first_subword_idxs = pad_sequences(idxs_to_collect_batch, maxlen=seq_length+2, dtype="long", truncating="post",padding="post")
    first_subword_idxs = torch.Tensor(first_subword_idxs).long()
    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1).long()

    doc_id = [2 for _ in doc_id] # deprecated
    doc_id = torch.Tensor(doc_id).unsqueeze(1).long()
    pos_tags = prepare_sequence(pos_tags, config.metadata.pos2id, -1, seq_length)
    pos_tags = pos_tags.cuda()
    doc_id, bert_tokens, trigger_tags, first_subword_idxs,\
    sent_lengths, bert_sentence_lengths = \
    doc_id.cuda(), bert_tokens.cuda(), trigger_tags.cuda(), first_subword_idxs.cuda(),\
    sent_lengths.cuda(), torch.Tensor(bert_sentence_lengths).cuda().long()

    # seed_trigger_idx = torch.Tensor(seed_trigger_idx).cuda()
    dataset = TensorDataset(doc_id, bert_tokens, trigger_tags, first_subword_idxs,
                            sent_lengths, bert_sentence_lengths, pos_tags)
    return dataset


def pair_trigger_maven_prompt(data_bert, prompt):
    # get trigger type set
    data_bert_new = []
    prompt = [prompt[x] for x in sorted(prompt.keys())]
    for i in range(len(prompt)):
        prompt[i] = prompt[i].split('-')
        if prompt[i][0] =='[CLS]':
            prompt[i] = prompt[i][1:]
        if prompt[i][-1] =='[SEP]':
            prompt[i] = prompt[i][:-1]
    for j in range(len(data_bert)):
        tokens, triggers, pos_tag, doc_id = data_bert[j]
        # tokens = config.tokenizer.tokenize(' '.join(tokens))
        query_sent = [['[CLS]'] + x + ['[SEP]'] + tokens + ['[SEP]'] for x in prompt]
        data_tuple = [(doc_id, tokens, qs, triggers, pos_tag) for qs in query_sent]
        data_bert_new.extend(data_tuple)

    return data_bert_new