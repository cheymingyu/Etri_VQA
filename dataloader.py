import os
import sys
import pickle
import pathlib
import json
import torch
import Constants
import numpy as np
import torch_geometric
import torchtext

ROOT_DIR = Constants.ROOT_DIR
SCENEGRAPHS = ROOT_DIR.joinpath('dataset', 'processed', 'sceneGraphs')
#EXPLAINABLE_GQA_DIR = ROOT_DIR.joinpath('GraphVQA')

SPLIT_TO_H5_PATH_TABLE = {
    'train_unbiased': '/home/pa5398/Experiments/SG_VQA/objectDetection/extract/save_train_feature.h5',
    'val_unbiased': '/home/pa5398/Experiments/SG_VQA/objectDetection/extract/save_test_feature.h5',
    'testdev': '/home/pa5398/Experiments/SG_VQA/objectDetection/extract/testdev_feature.h5',
    'debug': '/home/pa5398/Experiments/SG_VQA/objectDetection/extract/save_train_feature.h5',}

SPLIT_TO_MODE_TABLE = {
    'train_unbiased': 'train',
    'val_unbiased': 'test',
    'testdev': 'testdev',
    'debug': 'train',}

SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE = {
     'train_unbiased': str(ROOT_DIR / 'questions/train_balanced_programs.json'),
     'val_unbiased': str(ROOT_DIR / 'questions/val_balanced_programs.json'),
     'testdev': str(ROOT_DIR / 'questions/testdev_balanced_programs.json'),
     'debug': str(ROOT_DIR / 'debug_programs.json'),}
     #'val_all': str(ROOT_DIR / 'questions/val_all_programs.json')

class GQA_gt_sg_feature_lookup:
    SG_ENCODING_TEXT = torchtext.data.Field(sequential=True, tokenize="spacy",
                                            init_token="<start>", eos_token="<end>",
                                            include_lengths=False,
                                            tokenizer_language='en_core_web_sm',
                                            batch_first=False)

    def __init__(self, split):
        self.split = split
        assert split in SPLIT_TO_H5_PATH_TABLE

        self.bulid_scene_graph_encoding_vocab()

        if split == 'train_unbiased':
            with open(SCENEGRAPHS / 'train_sceneGraphs.json') as f:
                self.sg_json_data = json.load(f)
        else:
            assert split in ['val_unibased', 'testdev', 'val_all']
            if split == 'val_unbiased' or split == 'val_all':
                with open(SCENEGRAPHS / 'val_sceneGraphs.json') as f:
                    self.sg_json_data = json.load(f)
            elif split == 'testdev':
                self.sg_json_data = None

    def query_and_translate(self, queryID: str, new_execution_buffer: list):
        sg_this = self.sg_json_data[queryID]
        sg_datum = self.convert_one_gqa_scene_graph(sg_this)

        execution_bitmap = torch.zeros((sg_datum.num_nodes, GQATorchDataset.MAX_EXECUTION_STEP), dtype=torch.float32)
        instr_annotated_len = min(len(new_execution_buffer), GQATorchDataset.MAX_EXECUTION_STEP)
        padding_len = GQATorchDataset.MAX_EXECUTION_STEP - instr_annotated_len

        for instr_idx in range(instr_annotated_len):
            execution_target_list = new_execution_buffer[instr_idx]
            for trans_obj_id in execution_target_list:
                execution_bitmap[trans_obj_id, instr_idx] = 1.0

        for instr_idx in range(instr_annotated_len, instr_annotated_len + padding_len)
            execution_bitmap[:, instr_idx] = execution_bitmap[:, instr_annotated_len - 1]
        sg_datum.y = execution_bitmap

        return sg_datum

    def build_scene_graph_encoding_vocab(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        tmp_text_list = []
        tmp_text_list += load_str_list(ROOT_DIR / 'dataset/meta_info/name_gqa.txt')
        tmp_text_list += load_str_list(ROOT_DIR / 'dataset/meta_info/attr_gqa.txt')
        tmp_text_list += load_str_list(ROOT_DIR / 'dataset/meta_info/rel_gqa.txt')

        import Constants
        tmp_text_list += Constants.OBJECTS_INV + Constants.RELATIONS_INV + Constants.ATTRIBUTES_INV
        tmp_text_list.append("<self>")
        tmp_text_list = [tmp_text_list]

        GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT.build_vocab(tmp_text_list, vectors="glove.6B.300d")

        return GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT


    def convert_one_gqa_scene_graph(self, sg_this):
        if len(sg_this['objects'].keys()) == 0:
            sg_this = {
                'objects': {
                    '0': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '1',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },
                    '1': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '0',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },

                }
            }

        SG_ENCODING_TEXT = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT

        objIDs = sorted(sg_this['objects'].keys())
        map_objID_to_node_idx = {objID: node_idx for node_idx, objID in enumerate(objIDs)}

        node_feature_list = []
        edge_feature_list = []
        edge_topology_list = []
        added_sym_edge_list = []
        from_to_connections_set = set()

        for node_idx in range(len(objIDs)):
            objID = objIDs[node_idx]
            obj = sg_this['objects'][objID]
            for rel in obj['relations']:
                from_to_connections_set.add((node_idx, map_objID_to_node_idx[rel['object']]))

            MAX_OBJ_TOKEN_LEN = 12

            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int64) * SG_ENCODING_TEXT.vocab.stoi[SG_ENCODING_TEXT.pad_token]
            object_token_arr[0] = SG_ENCODING_TEXT.vocab.stoi[obj['name']]

            if object_token_arr[0] == 0: # ?????
                pass # ?????

            for attr_idx, attr in enumerate(set(obj['attributes'])):
                object_token_arr[attr_idx+1] = SG_ENCODING_TEXT.vocab.stoi[attr]
            node_feature_list.append(object_token_arr)
            # add self-loop
            edge_topology_list.append([node_idx, node_idx])
            edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi['<self>']], dtype=np.int64)
            edge_feature_list.append(edge_token_arr)

            for rel in obj['relations']:
                edge_topology_list.append([node_idx, map_objID_to_node_idx[rel["object"]]])
                edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi[rel['name']]], dtype=np.int64)
                edge_feature_list.append(edge_token_arr)
                # symmetric edge feature
                if (map_objID_to_node_idx[rel["object"]], node_idx) not in from_to_connections_set:
                    edge_topology_list.append(map_objID_to_node_idx[rel['object']], node_idx)
                    edge_feature_list.append(edge_token_arr)
                    added_sym_edge_list.append(len(edge_feature_list)-1)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        del edge_topology_list_arr

        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()

        datum = torch.geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        return datum

class GQATorchDataset(torch.utils.data.Dataset):
    MAX_EXECUTION_STEP = 5






