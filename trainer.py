import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from utils import *
import numpy as np
from losser import *
from tqdm import tqdm
import itertools
from torch_geometric.data import Data


class TrainingPreparation(object):
    def __init__(self, args):
        self.total_train_loss = {}
        self.total_valid_loss = {}

    def set_dataset(self, args):
        module = __import__('data_process.datasets', fromlist=[''])
        data_dir = args.data_dir
        tot_subj = 93
        test_subj = (tot_subj // args.k_folds) + 1
        subj_list = np.random.permutation(np.arange(tot_subj))
        subj_list = np.roll(subj_list, test_subj * args.test_fold)
        # train_subj = subj_list[test_subj:- test_subj][0:args.train_num].tolist()
        train_subj = subj_list[test_subj:- test_subj].tolist()
        valid_subj = subj_list[:test_subj].tolist()
        test_subj = subj_list[- test_subj:].tolist()
        self.trainset = module.Trainset(
            args,
            data_dir,
            subj_list=train_subj,
            method='S', ear='L',
            patch_range=args.p_range,
            n_samples=args.in_ch,
            sort_by_dist=False,
        )
        self.validset = module.Testset(
            args,
            data_dir,
            subj_list=valid_subj,
            method='S', ear='L',
            patch_range=args.p_range,
            n_samples=args.in_ch,
            mode='Valid',
            sort_by_dist=False,
        )
        self.testset = module.Testset(
            args,
            data_dir,
            subj_list=test_subj,
            method='S', ear='L',
            patch_range=args.p_range,
            n_samples=args.in_ch,
            mode='Test',
            sort_by_dist=True,
        )

    def set_gpu(self, args):
        self.train_loader = DataLoader(
            self.trainset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            drop_last=True
        )
        self.valid_loader = DataLoader(
            self.validset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )
        self.test_loader = DataLoader(
            self.testset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )

        module = __import__('networks.GCN2cat', fromlist=[''])

        in_channels = 129  # 节点特征维度
        hidden_channels = 64  # 隐藏层维度
        out_channels = 129  # 输出特征维度
        edge_dim = 7  # 边属性维度
        model_g_B = module.GATNet(
            in_channels_pos=28,
            hidden_channels=args.hidden_channels,
            heads=args.heads,
            out_channels=args.out_channels
        ).cuda()

        self.opt_g_B = torch.optim.Adam(model_g_B.parameters(), lr=args.g_lr, betas=(0.9, 0.999))

        self.scheduler_g_B = ExponentialLR(self.opt_g_B, gamma=args.alpha)

        self.model_g_B = model_g_B.cuda()

        self.lsd = LSD()
        self.valid_loss_lsd = np.inf
        self.valid_loss_lsd_cycle = np.inf
        self.valid_loss_mse = np.inf
        self.valid_loss_mse_cycle = np.inf
        self.valid_loss_sd = np.inf
        self.valid_global = np.inf
        self.valid_global_cycle = np.inf
        self.start_epoch = 0
        self.step = 0

    def save_checkpoint(self, args, epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "model": self.model_g_B.state_dict(),
            "optimizer": self.opt_g_B.state_dict(),
            "scheduler": self.scheduler_g_B.state_dict(),
            "epoch": epoch
        }
        checkpoint_path = os.path.join(checkpoint_dir, 'l2h_GAT_{}.pt'.format(epoch))
        torch.save(checkpoint_state, checkpoint_path)
        print("Saved checkpoint: {}".format(checkpoint_path))

    def train(self, args):
        seed = args.seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.model_g_B.train()
        LOSS = []
        lsd = LSD()
        # 定义损失函数
        adversarial_loss = nn.BCELoss()
        cycle_loss = nn.MSELoss()
        lambda_cycle = 10  # 循环一致性损失的权重
        root_dir = os.path.join('results/l2h_95/train', args.root_dirs)
        valid_error = np.inf
        best_loss = np.inf
        loss_best_lsd = np.inf
        loss_best_lsd_cy = np.inf
        loss_best_mse = np.inf
        loss_best_mse_cy = np.inf
        loss_best_global = np.inf
        loss_best_global_cy = np.inf
        print(f"单项L2H，训练集数量:{len(self.trainset) / 1730}")
        sum_lr = 1
        for epoch in range(self.start_epoch, args.total_epochs):
            loss_epoch = 0
            loss_lsd = 0
            loss_ga = 0
            loss_gb = 0
            loss_da = 0
            loss_db = 0
            loss_mse = 0
            loss_g = 0
            loss_lsd_cy = 0
            loss_mse_cy = 0
            loss_global = 0
            loss_global_cy = 0

            self.epoch = epoch
            for i, ts in enumerate(tqdm(self.train_loader, desc=f"第{epoch}轮")):
            # print(f"第{epoch}轮:train")
            # for i, ts in enumerate(self.train_loader):
                target, tar_pos, batch_data, an_mes, batch_datapos = ts
                target = target.cuda(non_blocking=True).float()
                tar_pos = tar_pos.cuda(non_blocking=True).float()
                batch_data = batch_data.cuda(non_blocking=True)
                batch_datapos = batch_datapos.cuda(non_blocking=True)

                an_mes = an_mes.cuda(non_blocking=True).float()

                target_m, target_p = logmag_phase(target)
                prediction = self.model_g_B(batch_data, batch_datapos)

                self.opt_g_B.zero_grad()
                loss_G = F.mse_loss(prediction, target_m).pow(0.5)
                loss_G.backward()

                self.opt_g_B.step()
                loss_g += loss_G
            if (loss_g / len(self.train_loader) < loss_best_mse):
                loss_best_mse = loss_g / len(self.train_loader)
            print(f"\ntrain:{loss_g / len(self.train_loader)}")
            if self.scheduler_g_B.get_last_lr()[0] > args.limit_lr:
                self.scheduler_g_B.step()
            else:
                sum_lr = sum_lr + 1
                if sum_lr % 3 ==0:
                    self.scheduler_g_B.step()
            print(f"lr:{self.scheduler_g_B.get_last_lr()}")
            # valid mode
            self.test(args, 'valid')
            self.model_g_B.train()
        print("\n")
        print("==" * 30)
        print(f"train:best mse:{loss_best_mse}")
        print(f"valid:best mse:{self.valid_loss_mse}")
        print(f"valid:best sd:{self.valid_loss_sd }")
        self.test(args, 'test')
        # self.save_checkpoint(args, self.epoch, root_dir)
        print("save final epoch")

    def test(self, args, mode='test'):
        self.model_g_B.eval()

        # 定义损失函数
        adversarial_loss = nn.BCELoss()
        cycle_loss = nn.MSELoss()
        lambda_cycle = 10  # 循环一致性损失的权重
        root_dir = root_dir = os.path.join('results/l2h_95/valid', args.root_dirs)
        os.makedirs(root_dir, exist_ok=True)
        loss_epoch = 0
        loss_lsd = 0
        loss_ga = 0
        loss_gb = 0
        loss_da = 0
        loss_db = 0
        loss_mse = 0
        loss_g = 0
        loss_sd = 0
        loss_lsd_cy = 0
        loss_mse_cy = 0
        loss_best_global = 0
        loss_best_global_cy = 0
        loss_global = 0
        loss_global_cy = 0

        loss_best_lsd = np.inf
        loss_best_lsd_cy = np.inf
        loss_best_mse = np.inf
        loss_best_mse_cy = np.inf
        with torch.no_grad():
            loader = self.valid_loader if mode == 'valid' else self.test_loader
            print(f"第{self.epoch}轮:valid")
            # for i, ts in enumerate(tqdm(loader,desc=f"第{self.epoch}轮测试开始")):
            for i, ts in enumerate(loader):
                target, tar_pos, batch_data, an_mes, batch_datapos = ts
                target = target.cuda(non_blocking=True).float()
                tar_pos = tar_pos.cuda(non_blocking=True).float()
                batch_data = batch_data.cuda(non_blocking=True)
                batch_datapos = batch_datapos.cuda(non_blocking=True)
                an_mes = an_mes.cuda(non_blocking=True).float()
                target_m, target_p = logmag_phase(target)
                prediction = self.model_g_B(batch_data, batch_datapos)
                loss_G = F.mse_loss(prediction, target_m).pow(0.5)
                loss_SD = (prediction - target_m).abs().float().squeeze(1).mean()

                loss_g += loss_G
                loss_sd += loss_SD

            if (mode == 'valid'):
                if (loss_g / len(loader) < self.valid_loss_mse):
                    self.valid_loss_mse = loss_g / len(loader)
                    self.valid_loss_sd = loss_sd / len(loader)
                    self.model_g_B.train()
                    # self.save_checkpoint(args,self.epoch,root_dir)
                    print("best Loss,save model as pt")

            print(f"\n{mode}:MSE:{loss_g / len(loader)}")
            print(f"\n{mode}:SD:{loss_sd / len(loader)}")
            print("==" * 30)
