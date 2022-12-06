import util
import numpy as np
import random
from transformers import AutoTokenizer
import os
from os.path import join
import json
import pickle
import logging
import torch
from bert_modelling import BertModel

logger = logging.getLogger(__name__)

class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            if self.config["dataset"] == "ontonotes":
                self.tensor_samples = {}
                tensorizer = Tensorizer(self.config)
                paths = {
                    'trn': join(self.data_dir, f'train.json'),
                    'dev': join(self.data_dir, f'dev.json'),
                    'tst': join(self.data_dir, f'test.json')
                }
                for split, path in paths.items():
                    logger.info('Tensorizing examples from %s; results will be cached)' % path)
                    is_training = (split == 'trn')
                    with open(path, 'r') as f:
                        samples = json.load(f)['data']
                    tensor_samples = tensorizer.tensorize_example(samples, is_training)
                    print(len(tensor_samples[0]))
                    self.tensor_samples[split] = [
                        self.convert_to_torch_tensor(*[x[i] for x in tensor_samples])
                    for i in range(len(tensor_samples[0]))]
                # print(self.tensor_samples)
                self.stored_info = tensorizer.stored_info
                # Cache tensorized samples
                with open(cache_path, 'wb') as f:
                    pickle.dump((self.tensor_samples, self.stored_info), f)
                

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, sentence_len, question_emb, is_training, gold_starts, gold_ends, gold_mention_cluster_map): # TODO
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        question_emb = torch.tensor(question_emb, dtype=torch.float)
        is_training = torch.tensor(is_training, dtype=torch.bool)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        
        return input_ids, input_mask, sentence_len, question_emb, is_training, gold_starts, gold_ends, gold_mention_cluster_map

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

    def get_stored_info(self): # TODO
        return self.stored_info

    def get_cache_path(self):
        if self.config["dataset"] == "ontonotes":
            cache_path = join(self.data_dir, f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
            
        return cache_path


class Tensorizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['bert_tokenizer_name'])

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}
        self.stored_info['constituents'] = {}
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # print('============', next(self.bert.parameters()).device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)
        # print('============', next(self.bert.parameters()).device)

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training): # TODO
        # Mentions and clusters
        max_sentence_len = self.config['max_segment_len']
        input_ids                = []
        input_mask               = []
        sentence_len             = []
        question                 = []
        gold_starts              = []
        gold_ends                = []
        gold_mention_cluster_map = []

        for data in example:
            print('data')
            for par in data['paragraphs']:
                print('par')
                tok = self.tokenizer(
                    # '[CLS]' + par['context'] + '[SEP]',
                    par['context'],
                    max_length=max_sentence_len,
                    truncation=True,
                    return_offsets_mapping=True
                )
                # print(tok)
                ids = tok['input_ids']
                slen = len(ids)
                msk = [1] * slen
                ids += [0] * (max_sentence_len - slen)
                msk += [0] * (max_sentence_len - slen)

                start_mp = {}
                end_mp = {}
                for i, (l, r) in enumerate(tok['offset_mapping']):
                    start_mp[l] = i
                    end_mp[r] = i + 1

                # print(start_mp, end_mp)
                for q in par['qas']:
                    que = self.tokenizer.tokenize(
                        q['question'],
                        max_length=max_sentence_len,
                        truncation=True
                    )
                    que = ['[CLS]'] + que + ['[SEP]']
                    que = que + ['[PAD]'] * (max_sentence_len - len(que))
                    que = self.tokenizer.convert_tokens_to_ids(que)

                    input_ids.append(ids)
                    input_mask.append(msk)
                    sentence_len.append(slen)
                    question.append(que)

                    golds = [(i['answer_start'], i['answer_start'] + len(i['text'].strip())) for i in q['answers']]
                    # print(golds)
                    for i in golds:
                        if not (i[0] in start_mp and i[1] in end_mp):
                            print('No match: ', par['context'], i[0], par['context'][i[0]:i[1]])
                    golds = [i for i in golds if i[0] in start_mp and i[1] in end_mp]
                    golds = [(start_mp[l], end_mp[r]) for l, r in golds]
                    if len(golds):
                        gold_start, gold_end = np.array(golds).T
                    else:
                        gold_start, gold_end = [], []

                    gold_starts.append(gold_start)
                    gold_ends.append(gold_end)
                    gold_mention_cluster_map.append([1] * len(golds))

        question = torch.tensor(question, dtype=torch.long)
        attn_mask = (question != 0).to(torch.long)
        seg_ids = torch.zeros(question.shape, dtype=torch.long)
        batch_size = 32
        question_emb = []
        for i in range((len(question) + batch_size-1) // batch_size):
            print(i * batch_size, len(question))
            s = slice(i * batch_size, (i+1) * batch_size)
            x = self.bert(question[s], attention_mask=attn_mask[s], token_type_ids=seg_ids[s])
            hidden_reps, cls_head = x[0], x[1]
            question_emb.append(cls_head.detach().cpu().numpy())

        question_emb = np.concatenate(question_emb)
        print(question_emb.shape, question_emb)

        is_training = [True] * len(input_ids)
        return (input_ids, input_mask, sentence_len, question_emb, is_training, gold_starts, gold_ends, gold_mention_cluster_map)


    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, coreferable_starts, coreferable_ends,
                         constituent_starts, constituent_ends, constituent_type,
                         max_sentences, sentence_offset=None):
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]
        
        coreferable_flags = (coreferable_starts < word_offset + num_words) & (coreferable_ends >= word_offset) if coreferable_starts is not None else None
        coreferable_starts = coreferable_starts[coreferable_flags] - word_offset if coreferable_starts is not None else None
        coreferable_ends = coreferable_ends[coreferable_flags] - word_offset if coreferable_starts is not None else None
        
        constituent_flags = (constituent_starts < word_offset + num_words) & (constituent_ends >= word_offset) if constituent_starts is not None else None
        constituent_starts = constituent_starts[constituent_flags] - word_offset if constituent_starts is not None else None
        constituent_ends = constituent_ends[constituent_flags] - word_offset if constituent_starts is not None else None
        constituent_type = constituent_type[constituent_flags] if constituent_type is not None else None

        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map, coreferable_starts, coreferable_ends, \
               constituent_starts, constituent_ends, constituent_type

if __name__ == '__main__':
    config = util.initialize_config('spanbert_large')
    x = CorefDataProcessor(config)