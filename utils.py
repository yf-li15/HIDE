import numpy as np
import torch
import pickle
import scipy.sparse as sp 
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def map_data(Data):
    s_data = Data[0]
    s_target = Data[1]
    cur_data = []
    cur_target = []
    for i in range(len(s_data)):
        data = s_data[i]
        target = s_target[i]
        if len(data) > 40:
            continue
        cur_data.append(data)
        cur_target.append(target)
    return [cur_data, cur_target]

def handle_data(inputData, sw, opt):
    items, len_data = [], []
    for nowData in inputData:
        len_data.append(len(nowData))
        Is = []
        for i in nowData:
            Is.append(i)
        items.append(Is)
    # len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)

    edge_lens = []
    for item_seq in items:
        item_num = len(list(set(item_seq)))
        num_sw = 0
        if opt.sw_edge:
            for win_len in sw:
                temp_num = len(item_seq) - win_len + 1
                num_sw += temp_num
        edge_num = num_sw
        if opt.item_edge:
            edge_num += item_num
        edge_lens.append(edge_num)

    max_edge_num = max(edge_lens)
    # reverse the sequence
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]

    #print(max_len, max_edge_num)

    return us_pois, us_msks, max_len, max_edge_num



class Data(Dataset):
    def __init__(self, data, all_train, opt, n_node, sw=[2]):
        self.n_node = n_node
        inputs, mask, max_len, max_edge_num = handle_data(data[0], sw, opt)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len # max_node_num
        self.max_edge_num = max_edge_num  # max_edge_num
        self.sw = sw # slice window
        self.opt = opt

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        max_n_node = self.max_len
        max_n_edge = self.max_edge_num # max hyperedge num

        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        if self.opt.sw_edge:
            for win in self.sw:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        if u_input[i+win-1] == 0:
                            break
                        for j in range(i, i+win):
                            rows.append(np.where(node == u_input[j])[0][0])
                            cols.append(edge_idx)
                            vals.append(1.0)
                        edge_idx += 1
        

        if self.opt.item_edge:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        if u_input[i] == item and i > 0:
                            rows.append(np.where(node == u_input[i-1])[0][0])
                            cols.append(edge_idx)
                            vals.append(2.0)
                    rows.append(np.where(node == item)[0][0])
                    cols.append(edge_idx)
                    vals.append(2.0)
                    edge_idx += 1
        
        # intent hyperedges are dynamic generated in layers.py
        u_Hs = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        Hs = np.asarray(u_Hs.todense())
        
        return [torch.tensor(alias_inputs), torch.tensor(Hs), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length
