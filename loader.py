import os
from tkinter import E
import numpy as np
import pandas as pd
import pickle
import logging
import torch
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=list(range(14)))
logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,args,tag,eval=False,inference=False):
        # tag MUST BE in {"train","dev","test"}
        self.max_len = args.max_seq_length
        self.inference = inference
        self.device = args.device
        self.model_type = args.model_type

        assert self.model_type in ["no_syntax","no_syntax_extra","with_chunking","with_const_tree","late_fusion"], "UNAVAILABLE model type. Possible options: no_syntax / with_chunking / with_const_tree"

        if self.model_type == "with_const_tree":
            data = pickle.load(open(os.path.join(args.data_dir,"const_tree",f"{tag}.pkl"),"rb"))
            if not inference:
                data = list(zip(data["wp_ids"],data["subtree_masks"],data["labels"]))
            else:
                data = list(zip(data["wp_ids"],data["subtree_masks"]))

        elif self.model_type == "late_fusion":
            data = pickle.load(open(os.path.join(args.data_dir,"late_fusion",f"{tag}.pkl"),"rb"))
            if not inference:
                data = list(zip(data["wp_ids"],data["dep_graphs"],data["labels"]))
            else:
                data = list(zip(data["wp_ids"],data["dep_graphs"]))
        else:
            data = pickle.load(open(os.path.join(args.data_dir,"chunking",f"{tag}.pkl"),"rb"))
            if not inference:
                if self.model_type in ["no_syntax","no_syntax_extra"]:
                    data = list(zip(data["wp_ids"],data["labels"]))
                else:
                    data = list(zip(data["wp_ids"],data["const_end_markers"],data["labels"]))
            else:
                if self.model_type in ["no_syntax","no_syntax_extra"]:
                    data = data["wp_ids"]
                else:
                    data = list(zip(data["wp_ids"],data["const_end_markers"]))
       
        if args.debug:
            data = data[:args.num_debug] 
        # shuffle the data for training set
        if not inference and not eval:
            np.random.seed(args.seed)
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            data = [data[i] for i in indices]

        data = [data[i:i+args.batch_size] for i in range(0,len(data),args.batch_size)]
        self.data = data
        logger.info(f"{tag}: {len(data)} batches generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]

        if self.model_type in ["no_syntax","no_syntax_extra"]:
            if self.inference:
                batch_wp_ids, batch_masks = self._padding(batch)
            else:
                batch_wp_ids, batch_labels = list(zip(*batch))
                batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
            encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_masks}
        elif self.model_type == "with_chunking":
            if self.inference:
                batch_wp_ids, batch_const_end_markers = list(zip(*batch))
            else:
                batch_wp_ids, batch_const_end_markers, batch_labels = list(zip(*batch))
            batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
            encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_masks,"wp2const":batch_const_end_markers}
        elif self.model_type == "with_const_tree":
            if self.inference:
                batch_wp_ids, batch_subtree_masks = list(zip(*batch))
            else:
                batch_wp_ids, batch_subtree_masks, batch_labels = list(zip(*batch))
            batch_wp_ids, batch_subtree_masks = self._padding(batch_wp_ids,batch_subtree_masks)
            encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_subtree_masks}
        elif self.model_type == "late_fusion":
            if self.inference:
                batch_wp_ids, batch_dep_graphs = list(zip(*batch))
            else:
                batch_wp_ids, batch_dep_graphs, batch_labels = list(zip(*batch))
            batch_wp_ids, batch_adj_mats = self._to_adj_mats(batch_wp_ids,batch_dep_graphs)
            encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_adj_mats}

        if not self.inference:
            batch_labels = mlb.fit_transform(batch_labels).astype(np.float32)
            encoding.update({"labels":torch.from_numpy(batch_labels).to(self.device)})
        return encoding

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _to_adj_mats(self,wp_ids,graphs):
        max_len = max(map(len,wp_ids))
        wp_ids = torch.Tensor([line + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        adj_mats = []
        for g in graphs:
            # assure that each wordpiece attend to itself; it is not included in the preprocessed dependency graph
            tmp_mat = np.eye(max_len)
            for k, v in g.items():
                tmp_mat[k,v] = 1
            adj_mats.append(tmp_mat)
        adj_mats = torch.Tensor(adj_mats).int().to(self.device)
        return wp_ids, adj_mats

    def _padding(self,wp_ids,masks=[]):
        max_len = max(map(len,wp_ids))
        wp_ids = torch.Tensor([line + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        if len(masks) == 0:
            padded_masks = torch.Tensor([len(line) * [1] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
            return wp_ids, padded_masks
        padded_masks = []
        for m in masks:
            m = {k:m[k] for k in sorted(m.keys())}
            tmp_mask = np.zeros((max_len,max_len))
            for k, v in m.items():
                if type(v) is list:
                    tmp_mask[k,v] = 1
                else:
                    assert type(v) is tuple, "ERROR: indexes of connected words must be represented in LIST or TUPLE."
                    tmp_mask[k,v[0]:v[1]] = 1
            padded_masks.append(tmp_mask)
        padded_masks = torch.Tensor(padded_masks).int().to(self.device)
        return wp_ids, padded_masks
