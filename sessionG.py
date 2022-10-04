import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from layers import DisentangleGraph, LocalHyperGATlayer
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse


class HIDE(Module):
    def __init__(self, opt, num_node, adj_all=None, num=None, cat=False):
        super(HIDE, self).__init__()
        # HYPER PARA
        self.opt = opt 
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.n_factor = opt.n_factor
        self.sample_num = opt.n_sample
        self.nonhybrid = opt.nonhybrid
        self.layer = int(opt.layer)
        self.n_factor = opt.n_factor     # number of intention prototypes
        self.cat = cat
        self.e = opt.e
        self.disen = opt.disen
        self.g = opt.g
        self.w_k = 10

        
        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)
        
        if self.disen:
            self.feat_latent_dim = self.dim // self.n_factor
            self.split_sections = [self.feat_latent_dim] * self.n_factor
            
        else:
            self.feat_latent_dim = self.dim
        
        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)


        if self.g:
            self.global_g = GlobalHyperGATlayer(opt, adj_all, num, embedding=self.embedding)
        
        if self.disen:
            self.disenG = DisentangleGraph(dim=self.feat_latent_dim, alpha=self.opt.alpha, e=self.e) # need to be updated
            self.disen_aggs = nn.ModuleList([LocalHyperGATlayer(self.feat_latent_dim, self.layer, self.opt.alpha, self.opt.dropout_gcn) for i in range(self.n_factor)])
        else:
            self.local_agg = LocalHyperGATlayer(self.dim, self.layer, self.opt.alpha, self.opt.dropout_gcn)



        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)
        
        
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()
        if self.disen:
            # define for the additional losses
            self.classifier = nn.Linear(self.feat_latent_dim,  self.n_factor)
            self.loss_aux = nn.CrossEntropyLoss()
            self.intent_loss = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_disentangle_loss(self, intents_feat):
        # compute discrimination loss
        
        labels = [torch.ones(f.shape[0])*i for i, f in enumerate(intents_feat)] # lable: 0, 1, ..., intent_num-1
        labels = trans_to_cuda(torch.cat(tuple(labels), 0)).long()
        intents_feat = torch.cat(tuple(intents_feat), 0)

        pred = self.classifier(intents_feat)
        discrimination_loss = self.loss_aux(pred, labels)
        return discrimination_loss

    def compute_scores(self, hidden, mask, item_embeddings):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        ht = hidden[:, 0, :]
        ht = ht.unsqueeze(-2).repeat(1, len, 1)             # (b, N, dim)
        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        

        hs = torch.cat([hs, ht], -1).matmul(self.w_s)

        feat = hs * hidden  
        nh = torch.sigmoid(torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        

        if self.disen:
            select = torch.sum(beta * hidden, 1)

            score_all = []
            select_split = torch.split(select, self.split_sections, dim=-1)
            b = torch.split(item_embeddings[1:], self.split_sections, dim=-1)
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]
                item_embeddings_int = b[i]
                scores_int = torch.mm(sess_emb_int, torch.transpose(item_embeddings_int, 1, 0))
                score_all.append(scores_int)
            
            score = torch.stack(score_all, dim=1)   # (b ,k, item_num)
            scores = score.sum(1)

        else:
            select = torch.sum(beta * hidden, 1)
            b = item_embeddings[1:]  # n_nodes x latent_size
            scores = torch.matmul(select, b.transpose(1, 0))

        return scores


    def forward(self, inputs, Hs, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        item_embeddings = self.embedding.weight
        
        #zeros = torch.cuda.FloatTensor(1, self.dim).fill_(0)
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embeddings = torch.cat([zeros, item_embeddings], 0)

        h = item_embeddings[inputs]
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(-1)
 
        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        session_c = session_c.unsqueeze(1)  # (batchsize, 1, dim)
        
        
        if self.disen:
            # intent prototypes from the clustering of all items
            all_items = item_embeddings[1:]   # item_num x dim
            intents_cat = torch.mean(all_items, dim=0, keepdim=True) # 1 x dim
            # Parallel disen-encoders
            mask_node = torch.ones_like(inputs)
            zeor_vec = torch.zeros_like(inputs)
            mask_node = torch.where(inputs.eq(0), zeor_vec, mask_node)
            
            h_split = torch.split(h, self.split_sections, dim=-1)
            s_split = torch.split(session_c, self.split_sections, dim=-1)
            intent_split = torch.split(intents_cat, self.split_sections, dim=-1)
            h_ints = []
            intents_feat = []
            for i in range(self.n_factor):
                h_int = h_split[i]
                Hs = self.disenG(h_int, Hs, intent_split[i], mask_node)  #  construct intent hyperedges for each item ?
                h_int = self.disen_aggs[i](h_int, Hs, s_split[i])

                # Activate disentangle with intent protypes
                # better 
                intent_p = intent_split[i].unsqueeze(0).repeat(batch_size, seqs_len, 1)
                # CosineSimilarity
                sim_val = h_int * intent_p
                cor_att = torch.sigmoid(sim_val)
                h_int = h_int * cor_att + h_int

                
                h_ints.append(h_int)
                intents_feat.append(torch.mean(h_int, dim=1))   # (b ,latent_dim)
                
           
            h_stack = torch.stack(h_ints, dim=2)   # (b ,len, k, latent_dim)
            h_local = h_stack.reshape(batch_size, seqs_len, self.dim)

            # Aux task: intent prediction
            self.intent_loss = self.compute_disentangle_loss(intents_feat)

        else:

            h_local = self.local_agg(h, Hs, session_c)
                        
        
        output = h_local
               
        return output, item_embeddings


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def forward(model, data):
    alias_inputs, Hs, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    Hs = trans_to_cuda(Hs).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden, item_embeddings = model(items, Hs, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    return targets, model.compute_scores(seq_hidden, mask, item_embeddings)




def train_test(model, train_data, test_data, top_K, opt):
    #print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        if opt.disen:
            loss += opt.lamda * model.intent_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    model.scheduler.step()

    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    
    for data in test_loader:
        targets, scores = forward(model, data)
        targets = targets.numpy()
        for K in top_K:
            sub_scores = scores.topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    return metrics
