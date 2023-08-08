import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np


class TriggerDetection(nn.Module):
    def __init__(self, config):
        super(TriggerDetection, self).__init__()
        self.config = config
        self.embedding_dim = config.EMBEDDING_DIM
        self.pretrained_weights = str(config.pretrained_weights)
        self.model = AutoModel.from_pretrained(self.pretrained_weights)
        self.linear = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.embedding_dim*2+config.metadata.pos_count, 2),
        )
        self.W = nn.Parameter(
            torch.rand(self.embedding_dim, self.embedding_dim) * 1. / np.sqrt(self.embedding_dim))
        self.model.config.output_attentions =True
        self.event_count = config.metadata.event_count
        self.last_k_hidden = config.last_k_hidden
        self.sqrt_d = np.sqrt(self.embedding_dim)
        self.cos = nn.CosineSimilarity(dim=-1)

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect_sent, idxs_to_collect_event):
        """
        Pick first subword embeddings with the indices list idxs_to_collect
        :param all_embeddings:
        :param idxs_to_collect:
        :param sentence_lengths:
        :return:
        """
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        # Other two mode need to be taken care of the issue
        # that the last index becomes the [SEP] after argument types
        sent_embeddings = []
        event_embeddings = []

        for i in range(N):
            to_collect_sent, to_collect_event = idxs_to_collect_sent[i], idxs_to_collect_event[i]
            to_collect_sent = to_collect_sent[to_collect_sent > 0]
            to_collect_event = to_collect_event[to_collect_event > 0]
            collected_sent = all_embeddings[i, to_collect_sent]  # collecting a slice of tensor
            collected_event = all_embeddings[i, to_collect_event]  # collecting a slice of tensor
            sent_embeddings.append(collected_sent)
            event_embeddings.append(collected_event)

        max_sent_len = idxs_to_collect_sent.shape[1]
        max_event_len = idxs_to_collect_event.shape[1]
        for i in range(N):
            sent_embeddings[i] = torch.cat((sent_embeddings[i], torch.zeros(max_sent_len - len(sent_embeddings[i]), self.embedding_dim).cuda()))
            event_embeddings[i] = torch.cat((event_embeddings[i], torch.zeros(max_event_len - len(event_embeddings[i]), self.embedding_dim).cuda()))

        sent_embeddings = torch.stack(sent_embeddings)
        event_embeddings = torch.stack(event_embeddings)
        return sent_embeddings, event_embeddings


    def forward(self, dataset_id, sentence_batch, idxs_to_collect_sent, idxs_to_collect_event, embed_lengths, pos_tag):
        # bert embeddings
        attention_mask = (sentence_batch!=0)*1
        token_type_ids = attention_mask.detach().clone()
        sep_idx = (sentence_batch == 102).nonzero(as_tuple=True)[1].reshape(sentence_batch.shape[0], -1)[:, 0]
        for i in range(sentence_batch.shape[0]):
            token_type_ids[i, :sep_idx[i]+1] = 0
        all_embeddings, _, hidden_layer_att = self.model(sentence_batch,
                                                        token_type_ids=token_type_ids,
                                                        attention_mask=attention_mask,
                                                        )

        # pick embeddings of sentence and event type
        sent_embeddings, event_embeddings = self.get_fist_subword_embeddings(all_embeddings,
                                                                             idxs_to_collect_sent,
                                                                             idxs_to_collect_event)
        # use BERT hidden logits to calculate contextual embedding
        # avg_layer_att = self.get_last_k_hidden_att(hidden_layer_att, idxs_to_collect_sent, self.last_k_hidden)
        # context_emb = avg_layer_att.matmul(sent_embeddings)

        map_numerator = 1.0/torch.sum(idxs_to_collect_event>0, dim=-1).float().unsqueeze(-1).unsqueeze(-1)
        logits = self.cos(sent_embeddings.unsqueeze(2).matmul(self.W), event_embeddings.unsqueeze(1))
        sent2event_att = logits.matmul(event_embeddings) * map_numerator

        pos_tag[pos_tag<0] = 16
        pos_tag = torch.nn.functional.one_hot(pos_tag, num_classes=17)
        # _logits = self.linear(torch.cat((sent_embeddings, sent2event_att, context_emb, pos_tag), dim=-1))

        _logits = self.linear(torch.cat((sent_embeddings, sent2event_att, pos_tag), dim=-1))
        return _logits

    def select_hidden_att(self, hidden_att, idxs_to_collect):
        """
        Pick attentions from hidden layers
        :param hidden_att: of dimension (batch_size, embed_length, embed_length)
        :return:
        """
        N = hidden_att.shape[0]
        sent_len = idxs_to_collect.shape[1]
        hidden_att = torch.mean(hidden_att, 1)
        hidden_att_selected = torch.zeros(N, sent_len, sent_len)
        if self.config.use_gpu:
            hidden_att_selected = hidden_att_selected.cuda()

        for i in range(N):
            to_collect = idxs_to_collect[i]
            to_collect = to_collect[to_collect>0]
            collected = hidden_att[i, to_collect][:,to_collect]  # collecting a slice of tensor
            hidden_att_selected[i, :len(to_collect), :len(to_collect)] = collected

        return hidden_att_selected/(torch.sum(hidden_att_selected, dim=-1, keepdim=True)+1e-9)

    def get_last_k_hidden_att(self, hidden_layer_att, idxs_to_collect, k=3):
        tmp = 0
        for i in range(k):
            tmp += self.select_hidden_att(hidden_layer_att[-i], idxs_to_collect)
        avg_layer_att = tmp/k
        return avg_layer_att
