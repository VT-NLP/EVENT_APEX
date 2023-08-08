import os
import torch
import json
import sys
sys.path.append(os.getenv('PROJECT_ROOT'))
sys.path.append(os.getcwd())
from utils_pre import pair_trigger_template, read_data_from, dataset_prepare_trigger_template,save_to_jsonl,save_trigger_dataset
from argparse import ArgumentParser
from utils import Config
from utils import Metadata



def prepare_trigger_template(config, file, few_trigger, event_rep, multiple_token_trigger=True):
    tokenizer = config.tokenizer
    word_to_ids = tokenizer.convert_tokens_to_ids

    # read data
    data_bert_train = read_data_from(file, tokenizer)

    data_bert_train = pair_trigger_template(data_bert_train, config, few_trigger, event_rep)
    data_loader = dataset_prepare_trigger_template(data_bert_train, tokenizer, config, word_to_ids,
                                                   config.metadata.triggers_to_ids)
    return data_loader


def get_event_rep(f='trigger_representation_template.json', rep='type_name_seed_template'):
    f = open(f, 'r')
    trigger_representation_json = json.load(f)
    f.close()
    return trigger_representation_json[rep]['suppliment_trigger']

def main(args):
    config = Config()
    torch.manual_seed(39)
    config.metadata = Metadata().ace
    event_types = sorted(config.metadata.trigger_set)

    test_types = [x for x in event_types if x.split(':')[0].lower() in {'life', 'personnel', 'transaction'} and x not in {'Justice:Acquit', 'Justice:Pardon', 'Justice:Extradite', 'Personnel:Nominate'}]
    test_type_ids = [config.metadata.triggers_to_ids[x] for x in test_types]
    train_types = [x for x in event_types if x.split(':')[0].lower() not in {'life', 'personnel', 'transaction'} and x not in {'Justice:Acquit', 'Justice:Pardon', 'Justice:Extradite', 'Personnel:Nominate'}]
    train_type_ids = [config.metadata.triggers_to_ids[x] for x in train_types]
    N = str(len(event_types))


    # load json files
    tr_json = json.load(open(args.sl_data_dir+args.prompts+'/train.json'))
    dev_json = json.load(open(args.sl_data_dir+args.prompts+'/dev.json'))
    te_json = json.load(open(args.sl_data_dir+args.prompts+'/test.json'))
    all_json = sum([te_json, dev_json, tr_json], [])
    novel_ex, seen_ex, mix_ex, non_ex = [], [], [], []
    for i, x in enumerate(all_json):
        if not x['event_trigger']:
            non_ex.append(i)
            continue
        events = {y[0] for y in x['event_trigger']}
        if events.intersection(set(test_types)) and not events.intersection(set(train_types)):
            novel_ex.append(i)
        elif not events.intersection(set(test_types)) and events.intersection(set(train_types)):
            seen_ex.append(i)
        elif events.intersection(set(test_types)) and events.intersection(set(train_types)):
            mix_ex.append(i)
        else:
            non_ex.append(i)
    #  re-split the datset for few-shot and zero-shot learning
    import random
    random.seed(39)
    random.shuffle(non_ex)
    random.shuffle(novel_ex)
    seen_novel_instances = {x: 0 for x in test_types}
    seen_novel_instances_id = {x: [] for x in test_types}
    for i in novel_ex:
        this_json = all_json[i]
        this_sent_type = list(set(x[0] for x in this_json['event_trigger']))
        if len(this_sent_type) == 1 and seen_novel_instances[this_sent_type[0]] + len(this_json['event_trigger']) <= args.few_shot_K:
            seen_novel_instances[this_sent_type[0]] += len(this_json['event_trigger'])
            seen_novel_instances_id[this_sent_type[0]].append(i)
    seen_novel_instances_id = sum(seen_novel_instances_id.values(), [])
    seen_ex = [seen_ex, seen_novel_instances_id]  # seen split and novel examples
    novel_ex = [x for x in novel_ex if x not in set(seen_novel_instances_id)]

    print('train base mentions: ' , len(sum([all_json[x]['event_trigger'] for x in seen_ex[0]], [])))
    print('train novel mentions: ' , len(sum([all_json[x]['event_trigger'] for x in seen_ex[1]], [])))
    print('dev mentions: ' , len(sum([all_json[x]['event_trigger'] for x in mix_ex], [])))
    print('test mentions: ' , len(sum([all_json[x]['event_trigger'] for x in novel_ex], [])))


    N_non_event = len(non_ex)
    N_train_ev = len(seen_ex[0]) + len(seen_ex[1])
    split_empty_ratio_tr = N_train_ev/(N_train_ev + len(novel_ex) + len(mix_ex))
    split_empty_ratio_dev = len(mix_ex)/(N_train_ev + len(novel_ex) + len(mix_ex))
    seen_ex[0].extend(non_ex[:int(len(non_ex)*split_empty_ratio_tr)])
    dev_split = non_ex[int(N_non_event*split_empty_ratio_tr) :
                            int(N_non_event*split_empty_ratio_tr) + int(N_non_event*split_empty_ratio_dev)] \
                + mix_ex
    te_split = non_ex[int(N_non_event*split_empty_ratio_tr) + int(N_non_event*split_empty_ratio_dev):] \
               + novel_ex

    # save to files
    save_to = args.out_dir + args.prompts + '/'
    # save split ids to json files
    with open(f'{args.out_dir}/splits/dev.json', 'w') as f:
        data = {'dev': mix_ex}
        json.dump(data, f)
    with open(f'{args.out_dir}/splits/test.json', 'w') as f:
        data = {'test': novel_ex}
        json.dump(data, f)
    with open(f'{args.out_dir}/splits/train.json', 'w') as f:
        data = {'base': seen_ex[0], 'novel': seen_ex[1]}
        json.dump(data, f)
    if args.few_shot:
        train_json_to_save = sum(seen_ex, [])
    elif args.zero_shot:
        train_json_to_save = seen_ex[0]
    for x, y in zip([te_split, dev_split, train_json_to_save], ['test', 'dev', 'train']):
        json_slit = [all_json[_] for _ in x]
        save_to_jsonl(json_slit, f'{args.out_dir}{args.prompts}/{y}.json')

    # read old date file
    te_dataset = torch.load(args.sl_data_dir+args.prompts+'/test.pt')
    dev_dataset = torch.load(args.sl_data_dir+args.prompts+'/dev.pt')
    tr_dataset = torch.load(args.sl_data_dir+args.prompts+'/train.pt')

    # revert tensordataset to list of objects
    all_data = []
    for data in [te_dataset, dev_dataset, tr_dataset]:
        this_dataset = []
        for x in range(9):
            this_dataset.append(torch.stack([data[y][x] for y in range(len(data))]))
        all_data.append(this_dataset)

    # combine dataset
    all_feature = []
    max_sec_dim = []
    for y in range(9):
        this_feature = []
        for x in range(3):
            this_feature.append(all_data[x][y])
        all_feature.append(this_feature)

    # prepare to combine train_dev_test
    for y in [0, 1, 2, 3, 6]:
        all_feature[y] = torch.cat(tuple(all_feature[y]), dim=0)
    max_sec_dim = []
    for y in [4, 5, 7, 8]:
        this_max_sec_dim = 0
        for x in range(3):
            this_max_sec_dim = max(this_max_sec_dim, all_feature[y][x].shape[1])
        max_sec_dim.append(this_max_sec_dim)

    all_y = [4, 5, 7, 8]
    pad_val = [0, 0, -1, 34]
    for i in range(len(all_y)):
        y = all_y[i]
        for x in range(3):
            a, b = all_feature[y][x].shape
            if b < max_sec_dim[i]:
                all_feature[y][x] = torch.cat((all_feature[y][x], pad_val[i] * torch.ones(a, max_sec_dim[i]-b).cuda()), dim=1)
        all_feature[y] = torch.cat(all_feature[y], dim=0)

    N = all_feature[0].shape[0]
    for i in range(9):
        all_feature[i] = all_feature[i].reshape(N//config.metadata.event_count, config.metadata.event_count, -1)

    for x, id in zip(['train'], [seen_ex]):
        data = []
        for feat in range(9):
            # add training examples for seen types
            this_seen_feat = all_feature[feat][id[0]]
            this_seen_feat = this_seen_feat[:, train_type_ids].flatten(end_dim=1)
            # add training examples for novel types
            if args.few_shot:
                this_novel_feat = all_feature[feat][id[1]]
                this_novel_feat = this_novel_feat[:, test_type_ids].flatten(end_dim=1)
                data.append(torch.cat([this_seen_feat, this_novel_feat]))
            elif args.zero_shot:
                data.append(this_seen_feat)
        save_path = save_to + x + '.pt'
        save_trigger_dataset(data, save_path)
    for x, id in zip(['dev'], [dev_split]):
        data = []
        for feat in range(9):
            this_feat = all_feature[feat][id]
            this_feat = this_feat[:, :]
            data.append(this_feat.flatten(end_dim=1))
        save_path = save_to + x + '.pt'
        save_trigger_dataset(data, save_path)
    for x, id in zip(['test'], [te_split]):
        data = []
        for feat in range(9):
            this_feat = all_feature[feat][id]
            this_feat = this_feat[:,test_type_ids]
            data.append(this_feat.flatten(end_dim=1))
        save_path = save_to + x + '.pt'
        save_trigger_dataset(data, save_path)

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create few-shot learning data for ace')
    arg_parser.add_argument('--out_dir', type=str,
                            default='',
                            help='Path to the output data folder')
    arg_parser.add_argument('--sl_data_dir', type=str,
                            default='',
                            help='Path to the original data folder')
    arg_parser.add_argument('--prompts', type=str,
                            default='APEX',
                            choices=['seed_trigger', 'event_name_definition', 'event_name_structure',
                                     'event_type_name', 'APEX'],
                            help='prompts')
    arg_parser.add_argument('--few_shot', type=bool,
                            default='',
                            help='few-shot learning setting')
    arg_parser.add_argument('--few_shot_K', type=int,
                            default=10,
                            help='few-shot number K')
    arg_parser.add_argument('--zero_shot', type=bool,
                            default='',
                            help='zero-shot setting')
    args = arg_parser.parse_args()
    main(args)
