import os
import torch
import json
import sys
from pathlib import Path
sys.path.append(os.getenv('PROJECT_ROOT'))
sys.path.append(os.getcwd())
from utils_pre import pair_trigger_template, read_data_from, dataset_prepare_trigger_template, save_to_json, save_trigger_dataset, get_event_rep, remove_irrelevent_data
from argparse import ArgumentParser
from utils import Config
from utils import Metadata
from torch.utils.data import TensorDataset


def prepare_trigger_template(config, file, few_trigger, event_rep, max_len=384):
    tokenizer = config.tokenizer
    word_to_ids = tokenizer.convert_tokens_to_ids

    # read data
    data_bert_raw = read_data_from(file, tokenizer, ace=False)
    data_bert_raw = [x for x  in data_bert_raw if len(x[1])< max_len]
    data_bert_train = pair_trigger_template(data_bert_raw[:], config, few_trigger, event_rep, ere=True)
    data_loader = dataset_prepare_trigger_template(data_bert_train, tokenizer, config, word_to_ids,
                                                   config.metadata.triggers_to_ids)


    return data_bert_raw, data_loader


def main(args):
    config = Config()
    config.update()
    config.train_file = args.data_dir + 'train.doc.txt'
    config.dev_file = args.data_dir + 'dev.doc.txt'
    config.test_file = args.data_dir + 'test.doc.txt'
    config.metadata = Metadata().ere
    event_types = sorted(config.metadata.trigger_set)

    # fetch and save to TensorDataset
    save_data_to = args.out_dir
    data = ['test', 'dev', 'train']
    files = [config.test_file, config.dev_file, config.train_file]
    json_path = 'trigger_prompts/trigger_representation_ere.json'
    for d, f in zip(data, files):
        event_rep = get_event_rep(json_path, rep=args.prompts)
        data_raw, d_loader = prepare_trigger_template(config, f, event_types, event_rep)
        save_path = f'{save_data_to}/ere_en/pt/sl/{args.prompts}/{d}.pt'
        save_trigger_dataset(d_loader, save_path)
        data = remove_irrelevent_data(data_raw[:], True)
        save_to_json(data, f'{save_data_to}/ere_en/pt/sl/{args.prompts}/{d}.json')

    return


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create supervised learning data for ere')
    arg_parser.add_argument('--out_dir', type=str,
                            default='',
                            help='Path to the output data folder')
    arg_parser.add_argument('--data_dir', type=str,
                            default='',
                            help='Path to the original data folder')
    arg_parser.add_argument('--splits_dir', type=str,
                            default='',
                            help='Path to the split folder')
    arg_parser.add_argument('--prompts', type=str,
                            default='APEX',
                            choices=['seed_trigger', 'event_name_definition', 'event_name_structure',
                                     'event_type_name', 'APEX'],
                            help='prompts')
    args = arg_parser.parse_args()
    main(args)