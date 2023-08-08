import json
from functools import partial
from argparse import ArgumentParser

def read_from_jsonl(path, save_sent_to=None):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    data = [json.loads(x) for x in json_list]
    if save_sent_to:
        f = open(save_sent_to, 'w')
        sents = sum([x['content'] for x in data], [])
        tokens = [' '.join(x['tokens']) for x in sents]
        f.write('\n'.join(tokens))
    return data


# read and write with pos_tag and doc_id
import spacy
from spacy.tokenizer import Tokenizer as spacy_tokenizer
import json
spacy_tagger = spacy.load("en_core_web_sm")
spacy_tagger.tokenizer = spacy_tokenizer(spacy_tagger.vocab)


def get_pos(sentence):
    doc = spacy_tagger(sentence)
    ret = []
    for token in doc:
        ret.append(token.pos_)
    return ret


def process_article(art, to_txt=False):
    title, idx, content, events, negative_triggers = \
        art['title'], art['id'], art['content'], art['events'], art['negative_triggers']
    tokens = [x['tokens'] for x in content]
    sents = [' '.join(x) for x in tokens]
    N = len(tokens)
    event_bios = [[] for _ in range(N)]
    for e in events:
        event_type, event_id, mentions = e['type'], e['type_id'], e['mention']
        for m in mentions:
            sent_id, offset = m['sent_id'], m['offset']
            sent_len = len(tokens[sent_id])
            this_event_bio = ['O'] * sent_len
            this_event_bio[offset[0]:offset[1]] = [event_type] * (offset[1]-offset[0])
            event_bios[sent_id].append(this_event_bio)
    for n in negative_triggers:
        sent_id, offset = n['sent_id'], n['offset']
        sent_len = len(tokens[sent_id])
        this_event_bio = ['O'] * sent_len
        this_event_bio[offset[0]:offset[1]] = ['Not'] * (offset[1]-offset[0])
        event_bios[sent_id].append(this_event_bio)
    pos_tags = list(map(get_pos, sents))
    if to_txt:
        ret = []
        for t, c, pos in zip(tokens, event_bios, pos_tags):
            txt = sum([[t], c, [pos]], [])
            txt = [' '.join(x) for x in txt]
            txt.append(idx)
            ret.append('\n'.join(txt))
        return ret

    return tokens, event_bios, pos_tags, idx


def process_article_wo_annotation(art, to_txt=False):
    title, idx, content, candidates = \
        art['title'], art['id'], art['content'], art['candidates']
    tokens = [x['tokens'] for x in content]
    sents = [' '.join(x) for x in tokens]
    N = len(tokens)
    event_bios = [[] for _ in range(N)]
    pos_tags = list(map(get_pos, sents))
    for c in candidates:
        sent_id, offset = c['sent_id'], c['offset']
        sent_len = len(tokens[sent_id])
        this_event_bio = ['O'] * sent_len
        this_event_bio[offset[0]:offset[1]] = ['Not'] * (offset[1]-offset[0])
        event_bios[sent_id].append(this_event_bio)

    if to_txt:
        ret = []
        for t, c, pos in zip(tokens, event_bios, pos_tags):
            txt = sum([[t], c, [pos]], [])
            txt = [' '.join(x) for x in txt]
            txt.append(idx)
            ret.append('\n'.join(txt))
        return ret

    return tokens, pos_tags, idx


def write_to(path, x):
    f = open(path, 'w')
    f.write(x)
    f.close()
    return


def main(args):
    # read test.jsonl from MAVEN dataset and save sentences to test_sent.txt, and save processed file to test.doc.txt
    # test set only defines trigger candidates
    for mode in ['test']:
        path = args.maven_path + mode + '.jsonl'
        data = read_from_jsonl(path, save_sent_to=args.out_dir + mode + '_sent.txt')

        processed_data = list(map(lambda x: process_article_wo_annotation(x, True), data))
        processed_data = sum(processed_data, [])
        processed_data = '\n\n'.join(processed_data)
        write_to(args.out_dir + mode + '.doc.txt', processed_data)

    # for training set and valid set
    # read .jsonl from MAVEN dataset and save sentences to _sent.txt, and save processed file to .doc.txt
    # both have event triggers and negative mentions
    for mode in ['valid', 'train']:
        path = args.maven_path + mode + '.jsonl'
        data = read_from_jsonl(path, save_sent_to=args.out_dir + mode + '_sent.txt')

        processed_data = list(map(lambda x: process_article(x, True), data))
        processed_data = sum(processed_data, [])
        processed_data = '\n\n'.join(processed_data)
        write_to(args.out_dir + mode + '.doc.txt', processed_data)
    return 0


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create supervised learning data for maven')
    arg_parser.add_argument('--maven_path', type=str,
                            default='',
                            help='Path to the original data')
    arg_parser.add_argument('--out_dir', type=str,
                            default='',
                            help='output folder')
    args = arg_parser.parse_args()
    main(args)


