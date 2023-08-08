import torch
import os
import sys
from data_to_dataloader_maven import data_prepare, read_data_maven, \
    pair_trigger_maven_prompt
sys.path.append(os.getenv('PROJECT_ROOT'))
sys.path.append(os.getcwd())
from utils_pre import save_to_json, get_event_rep
from argparse import ArgumentParser
from utils import Config
from utils import Metadata



def main(args):
    # configuration
    config = Config()
    config.update()
    config.train_file = args.data_dir + 'train.doc.txt'
    config.dev_file = args.data_dir + 'valid.doc.txt'
    config.test_file = args.data_dir + 'test.doc.txt'
    config.metadata = Metadata().maven
    event_types = sorted(config.metadata.trigger_set)

    # fetch and save to TensorDataset
    save_data_to = args.out_dir
    data = ['test', 'valid', 'train']
    files = [config.test_file, config.dev_file, config.train_file]
    json_path = 'trigger_prompts/trigger_representation_maven.json'
    event_rep = get_event_rep(json_path, rep=args.prompts)

    for d, f in zip(data, files):
        data_bert_raw = read_data_maven(f)
        save_path = f'{save_data_to}/sl/{args.prompts}/{d}.pt'
        data_bert_devel = pair_trigger_maven_prompt(data_bert_raw, event_rep)
        dev_loader = data_prepare(data_bert_devel, config)
        torch.save(dev_loader, save_path)
    return 0


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create supervised learning data for maven')
    arg_parser.add_argument('--out_dir', type=str,
                            default='',
                            help='Path to the output data folder')
    arg_parser.add_argument('--data_dir', type=str,
                            default='',
                            help='Path to the text data folder')
    arg_parser.add_argument('--prompts', type=str,
                            default='APEX',
                            choices=['seed_trigger', 'event_name_definition', 'event_name_structure',
                                     'event_type_name', 'APEX'],
                            help='prompts')

    args = arg_parser.parse_args()
    main(args)
