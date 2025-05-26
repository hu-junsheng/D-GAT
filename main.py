import argparse
import os
import torch
import numpy as np
from trainer import TrainingPreparation


def args_parse():
    parser = argparse.ArgumentParser()
    # ------------------------------
    # General
    # ------------------------------
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--plot_iter', type=int, default=500)
    parser.add_argument('--plot_train', action="store_true", help='plot train samples')
    parser.add_argument('--plot_test', action="store_true", help='plot test samples')
    parser.add_argument('--board_iter', type=int, default=500)
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--valid_epoch', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='directory path of ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of inference')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_lambda', type=float, default=0.5)
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--model', type=str, default='hyperfilm')
    parser.add_argument('--exp_name', type=str, default=None)
    # ------------------------------
    # DDP
    # ------------------------------
    parser.add_argument('--gpu_id', type=int, default=5,help='ID of the GPU to use (default: 0)')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--port', default='1235', type=str, help='port')
    # ------------------------------
    # Network
    # ------------------------------
    parser.add_argument('--cnn_layers', type=int, default=5)
    parser.add_argument('--rnn_layers', type=int, default=0)
    parser.add_argument('--in_ch', type=int, default=16)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--condition', type=str, default='hyper')
    parser.add_argument('--film_dim', type=str, default='chan')
    parser.add_argument('--without_anm', action='store_true')
    # ------------------------------
    # Data
    # ------------------------------
    parser.add_argument('--p_range', default=0.2, type=float, help='patch range')
    parser.add_argument('--rescale', default=50, type=float, help='rescale factor for input')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_dir', default='./HUTUBS/pkl-15', type=str)
    parser.add_argument("--mode",default="l2h",type=str)
    parser.add_argument("--root_dirs",default="l2h",type=str)
    parser.add_argument("--g_step", default=15, type=int)
    parser.add_argument("--g_lr", default=0.002, type=float)
    parser.add_argument("--wl", default=10, type=int)
    parser.add_argument("--wn", default=20, type=int)
    parser.add_argument("--da", default=100, type=int)
    parser.add_argument("--db", default=10, type=int)
    parser.add_argument("--cv", default=0.001, type=float)
    parser.add_argument("--db_scal", default=0.01, type=float)
    parser.add_argument("--Dopt", default=1, type=int)
    parser.add_argument("--nc", default=5, type=int)
    parser.add_argument("--hidden_channels", default=64, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--out_channels", default=129, type=int)
    parser.add_argument('--interpolation_default', action="store_true")
    parser.add_argument('--interpolation_Antipodal', action="store_true")
    parser.add_argument('--interpolation_samehorizontal', action="store_true")
    parser.add_argument('--interpolation_sameelevation', action="store_true")
    parser.add_argument("--newrange", default=12, type=float)
    parser.add_argument("--alpha", default=0.95, type=float)
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--lowrange", default=6.2, type=float)
    parser.add_argument("--limit_lr", default=9e-5, type=float)
    parser.add_argument("--train_num", default=19, type=int)
    parser.add_argument("--seeds", default=0, type=int)






    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = args_parse()
    seed = args.seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print("==============")
    print(f"args:{args}")
    tp = TrainingPreparation(args)
    tp.set_dataset(args)
    tp.set_gpu(args)
    tp.train(args)
    print("done")






