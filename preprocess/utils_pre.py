# coding=utf-8
import json
import os
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset

trigger_count = [0]


def _unpack_arg_wo_vt(data, tokenizer):
    """
    read the data with pos tags
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments = [], []
    if n > 4:
        mid = (n - 2) // 2
        triggers = [x.split(' ') for x in data[1:mid]]
        arguments = [x.split(' ') for x in data[mid:-3]]
        trigger_count[0] += len(triggers)

    entities = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, entities, pos, doc_id


def _unpack_ace_with_vt(data, tokenizer):
    """
    read the data with pos tags
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments = [], []
    if n > 6:
        mid = (n - 6) // 4 + 1
        triggers = [x.split(' ') for x in data[1:mid]]
        arguments = [x.split(' ') for x in data[mid:-5]]
        trigger_count[0] += len(triggers)

    entities = data[-5].split(' ')
    values = data[-4].split(' ')
    times = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, entities, values, times, pos, doc_id


def _unpack_ere_with_filler(data, tokenizer):
    """
    read the data with pos tags with fillers
    :param data:
    :param tokenizer:
    :return:
    """
    n = len(data)
    sentence = data[0].split()
    bert_words = tokenizer.tokenize(data[0])
    triggers, arguments, filler_args = [], [], []
    if n > 5:
        # mid = (n-5) // 3 + 1
        if (n - 5) % 3 != 0:
            data = split_multiple_event_roles(data)
        if (len(data) - 5) % 3 != 0:
            print('error in extracting data')
            pass
        else:
            mid = (n - 5) // 3 + 1
            triggers = [x.split(' ') for x in data[1:mid]]
            arguments = [x.split(' ') for x in data[mid:-4]]
            filler_args = arguments[1::2]
            arguments = arguments[::2]
            trigger_count[0] += len(triggers)

    entities = data[-4].split(' ')
    fillers = data[-3].split(' ')
    # values = data[-4].split(' ')
    # times = data[-3].split(' ')
    pos = data[-2].split(' ')
    doc_id = data[-1].split(' ')

    return sentence, bert_words, triggers, arguments, filler_args, entities, fillers, pos, doc_id


def read_data_from(file, tokenizer, ace=True, with_vt=True):
    """
    Extract data from ACE BIO files
    :param file: path
    :return: a list [[sentence1, triggers, arguments], [sentence2, triggers, arguments], ...]
                triggers' length is equal to arguments length, and each arguments row corresponding to the trigger
    """
    data = open(file, 'r').read().split('\n\n')
    output_ = [i.split('\n') for i in data]
    output = []

    # delete the last empty string in the data file if there is one
    if output_[-1] == ['']:
        output_ = output_[:-1]
    if ace and with_vt:
        output = list(map(partial(_unpack_ace_with_vt, tokenizer=tokenizer), output_))
    elif ace and not with_vt:
        output = list(map(partial(_unpack_arg_wo_vt, tokenizer=tokenizer), output_))
    elif not ace and with_vt:
        output = list(map(partial(_unpack_ere_with_filler, tokenizer=tokenizer), output_))
    elif ace and not with_vt:
        output = list(map(partial(_unpack_arg_wo_vt, tokenizer=tokenizer), output_))
    return output


def split_multiple_event_roles(data):
    ret = [data[0]]
    idxs = set()
    N, M = len(data) - 4, len(data[1].split(' '))
    for i in range(1, N):
        if '#@#' in data[i]:
            idxs.add(i)

    for i in range(1, N):

        if i not in idxs:
            ret.append(data[i])
        else:
            org_line = data[i].split(' ')
            new_split = []
            event_set = set(org_line)
            event_set.remove('O')
            event_set = list(event_set)[0][2:].split('#@#')
            for this_e in event_set:
                this_event = ['O' if org_line[k] == 'O' else org_line[k][0] + org_line[k][1] + this_e for k in range(M)]
                new_split.append(' '.join(this_event))
            ret.extend(new_split)
    ret.extend(data[-4:])
    return ret


def pair_trigger(data_bert, _event_args_dic, config, replace=False, replace_dic=None,
                 duplicate_positive=1, sample_negative=False):
    pos_count = 0
    # get trigger type set
    event_args_dic = _event_args_dic.copy()
    original_trigger_set = set(event_args_dic.keys())
    trigger_set = set(event_args_dic.keys())
    doc_id_dic = gen_doc_id()
    suplement_trigger = {}
    for tag_type in trigger_set:
        suplement_trigger[tag_type] = config.fact_container.suppliment_trigger[tag_type].split('-')
    if replace:
        for rep in replace_dic:
            trigger_set.remove(rep)
            trigger_set.add(replace_dic[rep])
            event_args_dic[replace_dic[rep]] = event_args_dic[rep]
            suplement_trigger[replace_dic[rep]] = suplement_trigger[rep]
    for x in suplement_trigger.keys():
        suplement_trigger[x] = config.tokenizer.tokenize(' '.join(suplement_trigger[x]))
    question_tags = {}

    for tag_type in trigger_set:
        question_tags[tag_type] = re.split('[: -]', tag_type.lower())
        question_tags[tag_type] = config.tokenizer.tokenize(' '.join(question_tags[tag_type]))
    data_bert_new = []
    i = 0
    original_trigger_set = sorted(list(original_trigger_set))
    for j in range(len(data_bert)):
        tokens, bert_words, event_bio, arg_bio, entity_bio, pos_tag, doc_id = data_bert[j]
        N = len(tokens)
        bert_sent_len = len(bert_words)
        bert_sent = ['[CLS]', '[EVENT]', '[SEP]'] + bert_words + ['[SEP]']

        event_queries = []
        event_tags = []
        # generate distinct instance for each trigger type w.r.t sentences
        for tag_type in original_trigger_set:
            if tag_type in replace_dic:
                question_tag_type = replace_dic[tag_type]
                # question_sup_type = config.suppliment_trigger[tag_type]
            else:
                question_tag_type = tag_type

            # supplementary triggers type tokens
            event_query = question_tags[question_tag_type][1:] + ['verb'] + \
                          suplement_trigger[question_tag_type] + ['[SEP]']

            has_event = False
            event_tags_this = ['O'] * N
            for k in range(len(event_bio)):
                for q in range(N):
                    if event_bio[k][q] == 'B-' + tag_type or event_bio[k][q] == 'I-' + tag_type:
                        event_tags_this[q] = event_bio[k][q][0]
                        has_event = True
                    if event_bio[k][q] == 'B-' + tag_type:
                        pos_count += 1

            event_queries.append(event_query)
            event_tags.append(event_tags_this)

        data_tuple = (doc_id_dic[doc_id[0]], tokens, bert_sent, event_queries, event_tags,
                      pos_tag, entity_bio)

        # sampling negative instance for training set
        if sample_negative and not has_event:
            if random.random() < sample_negative:
                data_bert_new.append(data_tuple)
        else:
            data_bert_new.extend([data_tuple] * duplicate_positive)

        i += 1
    return data_bert_new


def data_extract_trigger(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    sent_indx = [e[0] for e in data]
    words = [e[1] for e in data]
    bert_words = [e[2] for e in data]
    event_queries = [e[3] for e in data]
    event_tags = [e[4] for e in data]
    pos_tags = [e[5] for e in data]
    entity_bio = [e[6] for e in data]

    return sent_indx, words, bert_words, event_queries, event_tags, pos_tags, entity_bio


def prepare_bert_sequence(seq_batch, to_ix, pad, emb_len):
    padded_seqs = []
    for seq in seq_batch:
        pad_seq = torch.full((emb_len,), to_ix(pad), dtype=torch.int)
        # ids = [to_ix(w) for w in seq]
        ids = to_ix(seq)
        pad_seq[:len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def firstSubwordsIdx_for_one_seq(words, tokenizer, prompt=False):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    if not prompt:
        collected_1st_subword_idxs = []
        idx = 1  # need to skip the embedding of '[CLS]'
        has_seen_sep_at = 0
        for i in range(len(words)):

            w = words[i]
            if words[i] == '[SEP]':
                has_seen_sep_at = i
                collected_1st_subword_idxs.append(has_seen_sep_at)
                break

        idx = has_seen_sep_at + 1
        for i in range(has_seen_sep_at + 1, len(words)):
            w_tokenized = tokenizer.tokenize(words[i])
            collected_1st_subword_idxs.append(idx)
            idx += len(w_tokenized)
    else:
        collected_1st_subword_idxs = []
        idx = 1  # need to skip the embedding of '[CLS]'

        for i in range(1, len(words)):
            if words[i] == '[SEP]':
                break
            w_tokenized = tokenizer.tokenize(words[i])
            collected_1st_subword_idxs.append(idx)
            idx += len(w_tokenized)

    return collected_1st_subword_idxs


def firstSubwordsIdx_batch(seq_batch, tokenizer, prompt=False):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    for seq in seq_batch:
        idx_seq = firstSubwordsIdx_for_one_seq(seq, tokenizer, prompt)
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
            ids = [to_ix[w] if w in to_ix else -1 for w in seq]

        pad_seq[:len(ids)] = torch.Tensor(ids).long()
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)


def firstSubwordsIdx_for_one_seq_arg(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    collected_1st_subword_idxs = []
    idx = 1  # need to skip the embedding of '[CLS]'
    for i in range(1, len(words)):
        w = words[i]
        w_tokenized = tokenizer.tokenize(w)
        collected_1st_subword_idxs.append(idx)
        idx += len(w_tokenized)
        if w == '[SEP]':
            break

    bert_len = len(tokenizer.tokenize(' '.join(words)))
    collected_1st_subword_idxs.append(bert_len - 1)
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch_arg(seq_batch, tokenizer):
    """
    extract first subword indices for one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    idx_batch = []
    for seq in seq_batch:
        idx_seq = firstSubwordsIdx_for_one_seq_arg(seq, tokenizer)
        idx_batch.append(idx_seq)
    return idx_batch


def bio_to_ids(bio_tags, tags_to_ids, remove_overlap=False, is_trigger=False, is_entity=False):
    if remove_overlap:
        bio_tags = [[x.split('#@#')[0] for x in args] for args in bio_tags]
    if is_trigger:
        arg_remove_bio = [[tags_to_ids[x[2:]] if len(x) > 2 else tags_to_ids[x] for x in args] for args in bio_tags]
    elif is_entity:
        arg_remove_bio = [[tags_to_ids[re.split('[: -]', x[2:])[0]] if len(x) > 2 else tags_to_ids[x] for x in args] for
                          args in bio_tags]
    else:
        arg_remove_bio = [
            [tags_to_ids[re.split('[-#]', x[2:])[0]] if len(x) > 2 and x[2:] in tags_to_ids else tags_to_ids['O'] for x
             in args] for args in bio_tags]
    args = torch.Tensor(
        pad_sequences(arg_remove_bio, dtype="long", truncating="post", padding="post", value=tags_to_ids['[PAD]']))
    return args.long()


def gen_doc_id(train_test_split_path='$path_to_split_files'):
    """
    Generate a dictionary for document id dic
    :param train_test_split_path: path to split files
    :return: a dictionary for key being document and value being document name
    """
    ret_dic = dict()
    splits = ['train.txt', 'valid.txt', 'test.txt']
    j = 0
    for data in splits:
        files = open(train_test_split_path + data, 'r').read().splitlines()
        for i in range(len(files)):
            if '-kbp' in files[i]:
                files[i] = files[i][:-4]
            name = files[i].split('/')[-1] + '.csv'
            ret_dic[name] = i + j
        j += len(files)
    return ret_dic


def firstSubwordsIdx_for_one_seq_template(words, tokenizer):
    """
    extract first subword indices for one sentence
    :param words:
    :param tokenizer:
    :return:
    """
    each_token_len = [len(tokenizer.tokenize(w)) for w in words]
    collected_1st_subword_idxs = [sum(each_token_len[:x + 1]) for x in range(len(words) - 1) if words[x + 1] != '[SEP]']
    return collected_1st_subword_idxs


def firstSubwordsIdx_batch_template(seq_batch, tokenizer, is_template=True):
    """
    extract first subword indices temfor one batch
    :param seq_batch:
    :param tokenizer:
    :return:
    """
    all_idx_to_collect = [firstSubwordsIdx_for_one_seq_template(seq, tokenizer) for seq in seq_batch]
    all_sent = [' '.join(x[1:]) for x in seq_batch]
    if is_template:
        event_prefix_wo_def_len = [' '.join(x.split(' [SEP] ')[1:3]) for x in all_sent]
    else:
        event_prefix_wo_def_len = [' '.join(x.split(' [SEP] ')[1:]) for x in all_sent]
    event_prefix_wo_def_len = [len(x.split()) for x in event_prefix_wo_def_len]

    sent_len = [' '.join(x.split(' [SEP] ')[:1]) for x in all_sent]
    sent_len = [len(x.split()) for x in sent_len]

    sent_idx_to_collect = [all_idx_to_collect[x][:sent_len[x]] for x in range(len(all_idx_to_collect))]
    event_idx_to_collect = [all_idx_to_collect[x][sent_len[x]:sent_len[x] + event_prefix_wo_def_len[x]] for x in
                            range(len(all_idx_to_collect))]
    return sent_idx_to_collect, event_idx_to_collect


def pair_trigger_template(data_bert, config, event_types, event_template, ere=False,
                          data_split_file=os.getenv('SPLITS')):
    # doc_id_dic = gen_doc_id(data_split_file)
    data_bert_new = []

    for j in range(len(data_bert)):
        if ere:
            tokens, bert_words, event_bio, arg_bio, filler_bio, entity_bio, filler_ent_bio, pos_tag, doc_id = data_bert[
                j]
        else:
            tokens, bert_words, event_bio, arg_bio, entity_bio, value_bio, time_bio, pos_tag, doc_id = data_bert[j]

        trigger_arg_dic = trigger_arg_bio_to_ids(event_bio, arg_bio, event_types, len(entity_bio))

        for event_type in event_types:
            this_template = event_template[event_type].split('-')
            if this_template[0] == '[CLS]':
                this_template = this_template[1:]
            if this_template[-1] != '[SEP]':
                this_template.append('[SEP]')
            this_tokens = ['[CLS]'] + tokens + ['[SEP]'] + this_template
            this_bert_sent = config.tokenizer.tokenize(' '.join(this_tokens))

            this_trigger_bio = [x[0] for x in trigger_arg_dic[event_type]]
            this_ner_arg_bio = [x[1] for x in trigger_arg_dic[event_type]]

            data_tuple = (0, this_tokens, this_bert_sent, event_type, this_trigger_bio,
                          this_ner_arg_bio, pos_tag, entity_bio)

            data_bert_new.append(data_tuple)
    return data_bert_new


def dataset_prepare_trigger_template(data_bert, tokenizer, config, word_to_ix, trigger_to_ids, dataset_id=0,
                                     few_shot=1):
    """
    Generate data loader for the argument model
    :param data_bert:
    :param tokenizer:
    :param config:
    :param word_to_ix:
    :param trigger_to_ix:
    :return:
    """
    # unpack data
    doc_id, tokens, bert_sent, event_type, event_bio, ner_arg_bio, \
    pos_tag, entity_bio = data_extract_long_trigger(data_bert)

    # general information: sent_len, bert_sent_len, first_index

    idxs_to_collect_sent, idxs_to_collect_event = firstSubwordsIdx_batch_template(tokens, tokenizer)
    bert_sentence_lengths = [len(s) for s in bert_sent]
    max_bert_seq_length = int(max(bert_sentence_lengths))
    sentence_lengths = [len(x) for x in idxs_to_collect_sent]  # [x[-2]-1 for x in idxs_to_collect_batch]
    max_seq_length = int(max(sentence_lengths))
    bert_tokens = prepare_bert_sequence(bert_sent, word_to_ix, config.PAD_TAG, max_bert_seq_length)
    # general information: pad_sequence
    idxs_to_collect_sent = pad_sequences(idxs_to_collect_sent, dtype="long", truncating="post", padding="post")
    idxs_to_collect_sent = torch.Tensor(idxs_to_collect_sent)
    idxs_to_collect_event = pad_sequences(idxs_to_collect_event, dtype="long", truncating="post", padding="post")
    idxs_to_collect_event = torch.Tensor(idxs_to_collect_event)

    sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1)
    doc_id = torch.Tensor(doc_id).unsqueeze(1)
    pos_tags_all = prepare_sequence(pos_tag, config.metadata.pos2id, -1, max_seq_length)
    bert_sentence_lengths = torch.Tensor(bert_sentence_lengths)

    # trigger
    for i in range(len(event_bio)):
        if len(event_bio[i]) == 1:
            event_bio[i] = event_bio[i][0]
        else:
            event_bio[i] = [min(np.array(event_bio[i])[:, j]) for j in range(len(event_bio[i][0]))]
    event_tags = bio_to_ids(event_bio, trigger_to_ids, is_trigger=True)
    _sent_lengths = sent_lengths[::few_shot]

    n = doc_id.shape[0]
    dataset_id = dataset_id * torch.ones(n, 1)
    sent_ids_long = torch.Tensor(range(round(n / few_shot))).unsqueeze(-1).repeat(1, few_shot).reshape(-1, 1)

    long_data = (sent_ids_long, dataset_id,
                 sent_lengths, bert_sentence_lengths,
                 bert_tokens, idxs_to_collect_sent, idxs_to_collect_event, pos_tags_all, event_tags)
    return long_data


def data_extract_long_trigger(data, shuffle=False):
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)

    indices = index_array[:]
    data = [data[idx] for idx in indices]

    doc_id = [e[0] for e in data]
    words_batch = [e[1] for e in data]
    bert_words_batch = [e[2] for e in data]
    event_type = [e[3] for e in data]
    event_bio = [e[4] for e in data]
    new_arg_bio = [e[5] for e in data]
    # value_arg_bio = [e[6] for e in data]
    # time_arg_bio = [e[7] for e in data]
    pos_tag = [e[6] for e in data]
    entity_bio = [e[7] for e in data]

    return doc_id, words_batch, bert_words_batch, event_type, event_bio, new_arg_bio, \
           pos_tag, entity_bio


def trigger_arg_bio_to_ids(trigger_bio, arg_bio, event_type, sent_len):
    ret = defaultdict(list)

    # ner_arg = arg_bio[::3]
    # value_arg = arg_bio[1::3]
    # time_arg = arg_bio[2::3]
    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
            # ret[this_trigger].append([trigger_bio[i], ner_arg[i], value_arg[i], time_arg[i]])
            ret[this_trigger].append([trigger_bio[i], arg_bio[i]])

    no_this_trigger = ['O'] * sent_len
    for i in event_type:
        if not ret[i]:
            # ret[i] = [(no_this_trigger, no_this_trigger, no_this_trigger, no_this_trigger)]
            ret[i] = [(no_this_trigger, no_this_trigger)]

    return ret



def remove_irrelevent_data(data, ere=False):
    '''
    Keep input sentence, trigger annotations and argument annotations
    :param data:
    :param ere:
    :return:
    '''
    to_collect_idx = [0, 2, 3]
    data = [[x[y] for y in to_collect_idx] for x in data]
    for x in data:
        if not ere:
            x[2] = x[2][::3]  # discard value and time arguments
        if x[1] == []:
            x[1] = [['O' for _ in range(len(x[0]))]]
            x[2] = [['O' for _ in range(len(x[0]))]]
    return data


def get_event_rep(f='trigger_prompts/trigger_representation_ace.json', rep='type_name_seed_template'):
    f = open(f, 'r')
    trigger_representation_json = json.load(f)
    f.close()
    return trigger_representation_json[rep]['suppliment_trigger']


def save_to_json(data, file):
    res = []
    for x in data:
        event_list = []
        arg_list = []
        sentence, triggers, args = x
        sent_len = len(sentence)
        if set(triggers[0]) == {'O'}:
            res.append({'sentence': sentence, 'event_trigger': [], 'arg_list': []})
            continue
        for k in range(len(triggers)):
            trigger_ids = [i for i in range(sent_len) if triggers[k][i] != 'O']
            event_begin, event_end = trigger_ids[0], trigger_ids[-1] + 1
            event_type = triggers[k][event_begin][2:]
            arg_begins = [i for i in range(sent_len) if args[k][i][0] == 'B']
            arg_types = [args[k][i][2:] for i in range(sent_len) if args[k][i][0] == 'B']
            arg_ends = []
            for a in arg_begins:
                b = a + 1
                while b < sent_len:
                    if args[k][b][0] == 'I':
                        b += 1
                        continue
                    else:
                        break
                arg_ends.append(b)
            arg_list.extend([(event_type, x, y, z) for x, y, z in zip(arg_types, arg_begins, arg_ends)])
            event_list.append([event_type, event_begin, event_end])
        res.append({'sentence': sentence, 'event_trigger': event_list, 'arg_list': arg_list})
    jsonString = json.dumps(res)
    jsonFile = open(file, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    print('save to ', file)
    return res


def save_trigger_dataset(dataset, path=None):
    dataset = [x.cuda() for x in dataset]
    tensor_set = TensorDataset(*dataset)
    par_folder = os.path.dirname(path)

    if not os.path.exists(par_folder):
        Path(par_folder).mkdir(parents=True, exist_ok=True)
    torch.save(tensor_set, path)
    print('save file to ', path)
    return 0


def save_to_jsonl(json_output, path):
    with open(path, 'w') as outfile:
        for entry in json_output:
            json.dump(entry, outfile)
            outfile.write('\n')
    return


def pair_trigger_maven_prompt(data_bert, config, prompt, train=False):
    # get trigger type set
    data_bert_new = []

    for j in range(len(data_bert)):
        tokens, triggers, pos_tag, doc_id = data_bert[j]
        # tokens = config.tokenizer.tokenize(' '.join(tokens))
        query_sent = [['[CLS]'] + x + ['[SEP]'] + tokens + ['[SEP]'] for x in prompt]
        data_tuple = [(doc_id, tokens, qs, triggers, pos_tag) for qs in query_sent]
        data_bert_new.extend(data_tuple)

    return data_bert_new
