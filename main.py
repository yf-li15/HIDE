import sys
import time
import argparse
import pickle
import os
import logging
from sessionG import *
from utils import *



def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Nowplaying', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--model', default='HIDE', help='[GCEGNN, SRGNN, DHCN, SAHNN, COTREC]')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--w', type=int, default=6, help='max window size')
parser.add_argument('--n_factor', type=int, default=5, help='Disentangle factors number')
parser.add_argument('--gpu_id', type=str,default="1")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--n_iter', type=int, default=1)    
parser.add_argument('--seed', type=int, default=2021)                                 # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--e', type=float, default=0.4, help='Disen H sparsity.')
parser.add_argument('--disen', action='store_true', help='use disentangle')
parser.add_argument('--lamda', type=float, default=1e-4, help='aux loss weight')
parser.add_argument('--norm', action='store_true', help='use norm')
parser.add_argument('--g', action='store_true', help='use g')
parser.add_argument('--sw_edge', action='store_true', help='slide_window_edge')
parser.add_argument('--item_edge', action='store_true', help='item_edge')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id



def main():
    exp_seed = opt.seed
    top_K = [5, 10, 20]
    init_seed(exp_seed)

    sw = []
    for i in range(2, opt.w+1):
        sw.append(i)

    
    if opt.dataset == 'Tmall':
        num_node = 40727
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.0
        opt.e = 0.4
        opt.w = 6
        opt.nonhybrid = True
        sw = []
        for i in range(2, opt.w+1):
            sw.append(i)
        #sw=[2,3] # slide window

    elif opt.dataset == 'lastfm':
        num_node = 35231
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0

    
    else:
        num_node = 310

    print(">>SEED:{}".format(exp_seed))
    # ==============================
    print('===========config================')
    print("model:{}".format(opt.model))
    print("dataset:{}".format(opt.dataset))
    print("gpu:{}".format(opt.gpu_id))
    print("Disentangle:{}".format(opt.disen))
    print("Intent factors:{}".format(opt.n_factor))
    print("item_edge:{}".format(opt.item_edge))
    print("sw_edge:{}".format(opt.sw_edge))
    print("Test Topks{}:".format(top_K))
    print(f"Slide Window:{sw}")
    print('===========end===================')
   
    datapath = r'./datasets/'
    all_train = pickle.load(open(datapath + opt.dataset + '/all_train_seq.txt', 'rb'))
    train_data = pickle.load(open(datapath + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(datapath + opt.dataset + '/test.txt', 'rb'))


    train_data = Data(train_data, all_train, opt, n_node=num_node, sw=sw)
    test_data = Data(test_data, all_train, opt, n_node=num_node, sw=sw)

    if opt.model == 'HIDE':
        model = trans_to_cuda(HIDE(opt, num_node))
    start = time.time()

    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'EPOCH:{epoch}')
        print(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
        metrics = train_test(model, train_data, test_data, top_K, opt)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag = 1
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag = 1
        for K in top_K:
            print('Current Result:')
            print('\tP@%d: %.4f\tMRR%d: %.4f' %
                (K, metrics['hit%d' % K], K, metrics['mrr%d' % K]))
            print('Best Result:')
            print('\tP@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
