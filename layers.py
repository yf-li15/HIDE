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
        self.t = t              
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

        # CosineSimilarity 
        cos = nn.CosineSimilarity(dim=-1)
        sim_val = self.t * cos(hs, int_emb)                      # (batchsize, Node, Num_edge)
        
        
        sim_val = sim_val * mask
        
        # sort
        _, indices = torch.sort(sim_val, dim=1, descending=True)
        _, idx = torch.sort(indices, dim=1)

        # select according to <=0
        judge_vec = idx - select_k  
        ones_vec = 3*torch.ones_like(sim_val)
        zeros_vec = torch.zeros_like(sim_val)
        
        # intent hyperedges
        int_H = torch.where(judge_vec <= 0, ones_vec, zeros_vec)
        # add intent hyperedge
        H_out = torch.cat([int_H, H], dim=-1) # (batchsize, N, num_edge+1) 
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
            edge_cluster = torch.matmul(H_adj.transpose(1,2), h_emb)                  # (Batchsize, edge_num, latent_dim)
            h_t_cluster = h_emb + s_c
            
            # node2edge
            edge_c_in = edge_cluster.unsqueeze(1).expand(-1, N, -1, -1)               # (Batchsize, N, edge_num, latent_dim)
            h_4att0 = h_emb.unsqueeze(2).expand(-1, -1, edge_num, -1)                 # (Batchsize, N, edge_num, latent_dim)

            feat = edge_c_in * h_4att0

            atts10 = self.leakyrelu(torch.matmul(feat, self.a10).squeeze(-1))         # (Batchsize, N, edge_num)
            atts11 = self.leakyrelu(torch.matmul(feat, self.a11).squeeze(-1))         # (Batchsize, N, edge_num)
            atts12 = self.leakyrelu(torch.matmul(feat, self.a12).squeeze(-1))         # (Batchsize, N, edge_num)
            
            zero_vec = -9e15*torch.ones_like(H)
            alpha1 = torch.where(H.eq(1), atts10, zero_vec)
            alpha1 = torch.where(H.eq(2), atts11, alpha1)
            alpha1 = torch.where(H.eq(3), atts12, alpha1)

            alpha1 = F.softmax(alpha1, dim=1)                                         # (Batchsize, N, edge_num)

            edge = torch.matmul(alpha1.transpose(1,2), h_emb)                         # (Batchsize, edge_num, latent_dim)

            # edge2node
            edge_in = edge.unsqueeze(1).expand(-1, N, -1, -1)                         # (Batchsize, N, edge_num, latent_dim)
            h_4att1 = h_t_cluster.unsqueeze(2).expand(-1, -1, edge_num, -1)           # (Batchsize, N, edge_num, latent_dim)
            
            feat_e2n = edge_in * h_4att1
            
            atts20 = self.leakyrelu(torch.matmul(feat_e2n, self.a20).squeeze(-1))     # (Batchsize, N, edge_num)
            atts21 = self.leakyrelu(torch.matmul(feat_e2n, self.a21).squeeze(-1))     # (Batchsize, N, edge_num)
            atts22 = self.leakyrelu(torch.matmul(feat_e2n, self.a22).squeeze(-1))     # (Batchsize, N, edge_num)
            

            alpha2 = torch.where(H.eq(1), atts20, zero_vec)
            alpha2 = torch.where(H.eq(2), atts21, alpha2)
            alpha2 = torch.where(H.eq(3), atts22, alpha2)
            
            alpha2 = F.softmax(alpha2, dim=2)                                         # (Batchsize, N, edge_num)

            h_emb = torch.matmul(alpha2, edge)                                        # (Batchsize, N, latent_dim)
            h_embs.append(h_emb)

        h_embs = torch.stack(h_embs, dim=1)
        h_out = torch.sum(h_embs, dim=1)

        return h_out

