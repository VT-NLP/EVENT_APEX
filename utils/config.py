# coding=utf-8
import os

import torch
from transformers import AutoTokenizer


class Config(object):
    def __init__(self):
        self.project_root = os.getenv('PROJECT_ROOT')
        self.data_path = os.getenv('DATA_PATH')

        # file path
        self.tr_dataset = self.data_path + '/train.pt'
        self.te_dataset = self.data_path + '/test.pt'
        self.dev_dataset = self.data_path + '/dev.pt'
        self.dev_json = self.data_path + '/dev.json'
        self.te_json = self.data_path + '/test.json'
        self.tr_json = self.data_path + '/train.json'
        self.save_model_path = self.project_root + '/saved_models/'
        self.error_visualization_path = self.project_root + '/error_visualizations/'
        self.pretrain_model_id = ''
        self.pretrained_model_path = ''
        self.train_file_pt = ''
        self.dev_file_pt = ''
        self.test_file_pt = ''
        self.train_sent = ''
        self.dev_sent = ''
        self.test_sent = ''
        self.joint_train_pt = ''
        self.save_to_json_stats = ''
        self.event_rep = None
        self.log_file = ''
        self.PAD_TAG = '[PAD]'
        self.loss_type = 'sum'

        # device info
        self.use_gpu = True
        self.device = None
        self.torch_seed = 39

        # model parameters
        self.pretrained_weights = 'bert-large-uncased'
        self.tokenizer = None
        self.same_bert = False
        self.freeze_bert = False
        self.EMBEDDING_DIM = 1024
        self.ENTITY_EMB_DIM = 100
        self.extra_bert = -3
        self.use_extra_bert = False
        self.n_hid = 200
        self.dropout = 0.3
        self.do_train = True
        self.event_f1 = False
        self.load_pretrain = False
        self.last_k_hidden = 3
        self.prompt_lens = None
        self.trigger_threshold = 0.5
        self.shuffle = False

        # optimizer parameters
        self.trigger_training_weight = 1
        self.non_weight = 0.7
        self.train_batch_size = 32
        self.eval_batch_size = 33
        self.EPOCH = 5
        self.lr = 3e-5
        self.weight_decay = 0.01
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.eval_per_epoch = 10
        self.sampling = 0.25
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")

        # data details
        self.ere = False
        self.ace = False
        self.maven = False
        self.metadata = None
        self.few_shot = None
        self.zero_shot = None

    def set_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_weights)
            self.resize_tokenizer()
        except:
            raise ValueError("ValueError: pretrained_weights not set or not supported by AutoTokenizer. "
                             "Please set the tokenizer manually in config")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.set_tokenizer()

    def resize_tokenizer(self):
        special_tokens_dict = {'additional_special_tokens': ['<entity>', '</entity>', '<event>', '</event>', '[EVENT]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
