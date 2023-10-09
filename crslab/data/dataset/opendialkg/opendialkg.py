# @Time   : 2020/12/19
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/20, 2021/1/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE:
# @Time   : 2023/10/9
# @Author : Siyuan Lu
# @Email  : lusiyuanzs@gmail.com

r"""
OpenDialKG
==========
References:
    Moon, Seungwhan, et al. `"Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs."`_ in ACL 2019.

.. _`"Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs."`:
   https://www.aclweb.org/anthology/P19-1081/

"""

import json
import os
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class OpenDialKGDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        dpath = os.path.join(DATASET_PATH, 'opendialkg', tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        # self._load_vocab()
        self._load_other_data()

        vocab = {
            # 'tok2ind': self.tok2ind,
            # 'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'id2info': self.id2info,
            'id2entityid': self.id2entityid,
            # 'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data_id.json'), 'r', encoding='utf-8') as f:
            train_data = set(json.load(f))
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data_id.json')}]")
        with open(os.path.join(self.dpath, 'valid_data_id.json'), 'r', encoding='utf-8') as f:
            valid_data = set(json.load(f))
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data_id.json')}]")
        with open(os.path.join(self.dpath, 'test_data_id.json'), 'r', encoding='utf-8') as f:
            test_data = set(json.load(f))
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data_id.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # opendialkg
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8')) # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = open(os.path.join(self.dpath, 'kg.json'), encoding='utf-8')
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'kg.json')}]")

        # conceptnet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {concept \t relation\t concept}
        self.word_kg = open(os.path.join(self.dpath, 'concept_subkg.txt'), encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'word2id.json')} and {os.path.join(self.dpath, 'concept_subkg.txt')}]")

        self.id2info = json.load(open(os.path.join(self.dpath, 'id2info.json'), 'r', encoding='utf-8'))
        self.id2name = {}
        self.id2entityid = {}
        for id, info in self.id2info.items():
            self.id2name[id] = info['name']
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]
        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'id2info.json')}]")

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data, processed_valid_data, processed_test_data = self._raw_data_process(train_data, valid_data, test_data)
        logger.debug("[Finish raw data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, train_data, valid_data, test_data):
        augmented_train_data = []
        augmented_valid_data = []
        augmented_test_data = []
        augmented_convs = []

        with open(os.path.join(self.dpath, 'data.jsonl'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                dialog = json.loads(line)
                dialog_turn_id, augmented_convs = self._merge_conv_data(dialog, train_data, valid_data, test_data)
                if dialog_turn_id in train_data:
                    augmented_train_data.append(augmented_convs)
                if dialog_turn_id in valid_data:
                    augmented_valid_data.append(augmented_convs)
                if dialog_turn_id in test_data:
                    augmented_test_data.append(augmented_convs)

        augmented_train_dicts = []
        augmented_valid_dicts = []
        augmented_test_dicts = []

        for conv in tqdm(augmented_train_data):
            augmented_train_dicts.extend(self._augment_and_add(conv))
        for conv in tqdm(augmented_valid_data):
            augmented_valid_dicts.extend(self._augment_and_add(conv))
        for conv in tqdm(augmented_test_data):
            augmented_test_dicts.extend(self._augment_and_add(conv))

        return augmented_train_dicts, augmented_valid_dicts, augmented_test_dicts

    def _merge_conv_data(self, conversation, train_data, valid_data, test_data):
        augmented_convs = []
        last_role = None
        for utt in conversation['dialog']:
            role = utt['role']
            text = utt['text']
            entity_turn = utt['entity']
            item_turn = utt['item']

            item_ids = [self.entity2id[movie] for movie in utt['item'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            dialog_turn_id = str(conversation['dialog_id']) + '_' + str(utt['turn_id'])

            if role == 'assistant':
                role = 'Recommender'
            elif role == 'user':
                role = 'User'

            if role == last_role:
                augmented_convs[-1]["text"] += text
                augmented_convs[-1]["item"] += item_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:
                augmented_convs.append({
                    "dialog_id": dialog_turn_id,
                    "role": role,
                    "text": text,
                    "entity": entity_ids,
                    "item": item_ids,
                })
            last_role = role

            if dialog_turn_id in train_data or dialog_turn_id in valid_data or dialog_turn_id in test_data:
                return dialog_turn_id, augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context, context_entities, context_words, context_items = [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            # text_tokens, entities, items, words = conv["text"], conv["entity"], conv["item"], conv["word"]
            text, entities, items = conv["text"], conv["entity"], conv["item"]
            if len(context) > 0:
                conv_dict = {
                    "dialog_id": conv['dialog_id'],
                    'role': conv['role'],
                    "context": copy(context),
                    "response": text,
                    "context_entities": copy(context_entities),
                    # "context_words": copy(context_words),
                    'context_items': copy(context_items),
                    "items": items,
                }
                augmented_conv_dicts.append(conv_dict)

            context.append(text)
            context_items += items
            for entity in entities + items:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            # for word in words:
            #     if word not in word_set:
            #         word_set.add(word)
            #         context_words.append(word)

        return augmented_conv_dicts

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")
        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        item_entity_ids = json.load(open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load item entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": item_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
        edge_list = []  # [(entity, entity, relation)]
        for line in self.entity_kg:
            triple = line.strip().split('\t')
            if len(triple) != 3 or triple[0] not in self.entity2id or triple[2] not in self.entity2id:
                continue
            e0 = self.entity2id[triple[0]]
            e1 = self.entity2id[triple[2]]
            r = triple[1]
            edge_list.append((e0, e1, r))
            # edge_list.append((e1, e0, r))
            edge_list.append((e0, e0, 'SELF_LOOP'))
            if e1 != e0:
                edge_list.append((e1, e1, 'SELF_LOOP'))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 20000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])

        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }

    def _word_kg_process(self):
        edges = set()  # {(entity, entity)}
        entities = set()
        for line in self.word_kg:
            triple = line.strip().split('\t')
            entities.add(triple[0])
            entities.add(triple[2])
            e0 = self.word2id[triple[0]]
            e1 = self.word2id[triple[2]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }
    
    def get_attr_list(self):
        attr_list = ['genre', 'actor', 'director', 'writer']
        return attr_list

    def get_ask_instruction(self):
        ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: ask my preference for writer
E: I can directly give recommendations
You have selected {}, do not repeat them. Please enter the option character.'''
        option2attr = {
            'A': 'genre',
            'B': 'actor',
            'C': 'director',
            'D': 'writer',
            'E': 'recommend'
        }
        option2template = {
            'A': 'Which genre do you like?',
            'B': 'Which actor do you like?',
            'C': 'Which director do you like?',
            'D': 'Which writer do you like?',
        }

        rec_instruction = 'Please give me 10 recommendations according to my preference (Format: no. title. No other things except the item list in your response). You can recommend mentioned items in our dialog.'

        ask_instruction_dict = {
            'ask_instruction': ask_instruction,
            'option2attr': option2attr,
            'option2template': option2template,
            'rec_instruction': rec_instruction
        }

        return ask_instruction_dict