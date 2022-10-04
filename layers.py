from numpy.lib.function_base import _DIMENSION_NAME
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import math


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
    
class DisentangleGraph(nn.Module):
    def __init__(self, dim, alpha, e=0.3, t=10.0):
        super(DisentangleGraph, self).__init__()
        # Disentangling Hypergraph with given H and latent_feature
        self.latent_dim = dim   # Disentangled feature dim
        self.e = e              # sparsity parameters
        self.t = t              # 1/tao
        self.w = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.w1 = nn.Parameter(torch.Tensor(self.latent_dim, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, H, int_emb, mask):
        """
        Input: intent-aware hidden:(Batchsize, N, dim), incidence matrix H:(batchsize, N, num_edge), intention_emb: (num_factor, dim), node mask:(batchsize, N)
        Output: Distangeled incidence matrix
        """
        node_num = torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1) # (batchsize, 1, 1)
        select_k = self.e * node_num
        select_k = select_k.floor() 

        mask = mask.float().unsqueeze(-1) # (batchsize, N, 1)
        h = hidden
        batch_size = h.shape[0]
        N = H.shape[1]
        k = int_emb.shape[0]

        select_k = select_k.repeat(1, N, k)

          
        int_emb =  int_emb.unsqueeze(0).repeat(batch_size, 1, 1) # (batchsize, num_factor, latent_dim)
        int_emb =  int_emb.unsqueeze(1).repeat(1, N, 1, 1)       # (batchsize, N, num_factor, latent_dim)

        hs = h.unsqueeze(2).repeat(1, 1, k, 1)                   # (batchsize, N, num_factor, latent_dim)

        '''
        feat = int_emb * hs

        sim_val = torch.sigmoid(torch.matmul(feat, self.w1).squeeze(-1))         # (Batchsize, N, num_factor) ?remove map
        # sim_val = self.leakyrelu(torch.matmul(feat, self.w1).squeeze(-1))
        '''

      
        # CosineSimilarity 
        cos = nn.CosineSimilarity(dim=-1)
        sim_val = self.t * cos(hs, int_emb)     # [0, 1]   # (batchsize, Node, Num_edge)
        
        
        sim_val = sim_val * mask
        
        # sort
        _, indices = torch.sort(sim_val, dim=1, descending=True)
        _, idx = torch.sort(indices, dim=1) # important

        judge_vec = idx - select_k # select according to <=0

        ones_vec = 3*torch.ones_like(sim_val)
        zeros_vec = torch.zeros_like(sim_val)
        # intent hyperedges
        int_H = torch.where(judge_vec <= 0, ones_vec, zeros_vec)
  
     
        H_out = torch.cat([int_H, H], dim=-1) # (batchsize, N, num_edge+1) add intent hyperedge
        # return learned binary value
        return H_out


class LocalHyperGATlayer(nn.Module):
    def __init__(self, dim, layer, alpha, dropout=0., bias=False, act=True):
        super(LocalHyperGATlayer, self).__init__()
        self.dim = dim
        self.layer = layer
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.act = act

        if self.act:
            self.acf = torch.relu

        
        # Parameters 
        # node->edge->node
        self.w1 = Parameter(torch.Tensor(self.dim, self.dim))
        self.w2 = Parameter(torch.Tensor(self.dim, self.dim))

        # In hyperedge, Out hyperedge, sw Hyperedge   
        self.a10 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))   
        self.a11 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))   
        self.a12 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))    
        self.a20 = nn.Parameter(torch.Tensor(size=(self.dim, 1))) 
        self.a21 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))     
        self.a22 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))  
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, hidden, H, s_c):
        """
        Input: hidden:(Batchsize, N, latent_dim), incidence matrix H:(batchsize, N, num_edge), session cluster s_c:(Batchsize, 1, latent_dim)
        Output: updated hidden:(Batchsize, N, latent_dim)
        """
        batch_size = hidden.shape[0]
        N = H.shape[1]            # node num
        edge_num = H.shape[2]     # edge num
        H_adj = torch.ones_like(H)
        mask = torch.zeros_like(H)
        H_adj = torch.where(H>0, H_adj, mask)
        s_c = s_c.expand(-1, N, -1)
        h_emb = hidden
        h_embs = []

        for i in range(self.layer):
            edge_cluster = torch.matmul(H_adj.transpose(1,2), h_emb)       # (Batchsize, edge_num, latent_dim)
            # h_t_cluster = torch.matmul(H_adj, edge_cluster) + s_c          #?
            h_t_cluster = h_emb + s_c
            # h_t_cluster = torch.matmul(H_adj, edge_cluster) + s_c +  h_emb
            
            # node2edge
            
            edge_c_in = edge_cluster.unsqueeze(1).expand(-1, N, -1, -1)           # (Batchsize, N, edge_num, latent_dim)
            h_4att0 = h_emb.unsqueeze(2).expand(-1, -1, edge_num, -1)             # (Batchsize, N, edge_num, latent_dim)

            feat = edge_c_in * h_4att0

            
            atts10 = self.leakyrelu(torch.matmul(feat, self.a10).squeeze(-1))         # (Batchsize, N, edge_num)
            atts11 = self.leakyrelu(torch.matmul(feat, self.a11).squeeze(-1))         # (Batchsize, N, edge_num)
            atts12 = self.leakyrelu(torch.matmul(feat, self.a12).squeeze(-1))         # (Batchsize, N, edge_num)
            
            zero_vec = -9e15*torch.ones_like(H)
            alpha1 = torch.where(H.eq(1), atts10, zero_vec)
            alpha1 = torch.where(H.eq(2), atts11, alpha1)
            alpha1 = torch.where(H.eq(3), atts12, alpha1)

            alpha1 = F.softmax(alpha1, dim=1) # (Batchsize, N, edge_num)

            edge = torch.matmul(alpha1.transpose(1,2), h_emb) # (Batchsize, edge_num, latent_dim)

            
            # edge2node
            edge_in = edge.unsqueeze(1).expand(-1, N, -1, -1)           # (Batchsize, N, edge_num, latent_dim)
            h_4att1 = h_t_cluster.unsqueeze(2).expand(-1, -1, edge_num, -1)  # (Batchsize, N, edge_num, latent_dim)
            
            feat_e2n = edge_in * h_4att1
            
            atts20 = self.leakyrelu(torch.matmul(feat_e2n, self.a20).squeeze(-1))         # (Batchsize, N, edge_num)
            atts21 = self.leakyrelu(torch.matmul(feat_e2n, self.a21).squeeze(-1))         # (Batchsize, N, edge_num)
            atts22 = self.leakyrelu(torch.matmul(feat_e2n, self.a22).squeeze(-1))         # (Batchsize, N, edge_num)
            

            alpha2 = torch.where(H.eq(1), atts20, zero_vec)
            alpha2 = torch.where(H.eq(2), atts21, alpha2)
            alpha2 = torch.where(H.eq(3), atts22, alpha2)
            
            alpha2 = F.softmax(alpha2, dim=2) # (Batchsize, N, edge_num)

            h_emb = torch.matmul(alpha2, edge) # (Batchsize, N, latent_dim)
            h_embs.append(h_emb)

        h_embs = torch.stack(h_embs, dim=1)
        #print(embs.size())
        h_out = torch.sum(h_embs, dim=1)

        return h_out


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output


class GlobalHyperGATlayer(nn.Module):
    def __init__(self, opt, adj_all, num, embedding):
        super(GlobalHyperGATlayer, self).__init__()
        self.dim = opt.hiddenSize
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.embedding = embedding
        # Aggregator
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        
    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    

    def forward(self, inputs, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        return h_global
        





