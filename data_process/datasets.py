import torch
import numpy as np
from torch.utils import data
import os
import glob
import random
from tqdm import tqdm
import logging
from utils import *
import sofa
from torch_geometric.data import Data


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, args, directory, subj_list, method='S', ear='L', patch_range=0.2, n_samples=1,
                 sort_by_dist=False):
        np.random.seed(0)
        self.method = method  # method \in ['S', 'M']
        self.ear = ear  # ear \in ['L', 'R']
        self.data_path = sorted(glob.glob(os.path.join(directory, '*.pkl')))
        self.num_grid = 1730 if method == 'S' else 440
        self.files = subj_list
        self.pos = 's_pos' if method == 'S' else 'm_pos'
        self.range = patch_range
        self.n_samples = n_samples
        self.sort_by_dist = sort_by_dist
        self.sigma = args.sigma
        self.lowrange = args.lowrange
        if args.interpolation_default:
            print("按距离插值")
            patch_range = args.newrange
            if os.path.exists(f"nbhd/nbhd-{patch_range}.pkl"):
                self.nbhd = load_dict(f"nbhd/nbhd-{patch_range}.pkl")
                print(f"... loaded nbhd dict of range {patch_range}")
            else:
                os.makedirs("nbhd", exist_ok=True)
                self.prep_nbhd()
                save_dict(self.nbhd, f"nbhd/nbhd-{patch_range}.pkl")
        else:
            print("按角度插值")
            self.range = args.newrange
            if os.path.exists(f"nbhd/nbhd_new-{self.range}.pkl"):
                self.nbhd = load_dict(f"nbhd/nbhd_new-{self.range}.pkl")
                print(f"... loaded nbhd_new dict of range {self.range}")
            else:
                os.makedirs("nbhd", exist_ok=True)
                self.prep_nbhdnew()
                save_dict(self.nbhd, f"nbhd/nbhd_new-{self.range}.pkl")

        self.data = {}
        for data_idx in self.files:
            data_tgt = self.data_path[data_idx]
            self.data[data_idx] = load_dict(data_tgt)

    def __len__(self):
        return self.num_grid * len(self.files)

    def __getitem__(self, index):

        subj_idx = int(index // self.num_grid)
        tgts_idx = int(index % self.num_grid)
        data_idx = self.files[subj_idx]
        hrirs = self.data[data_idx]['feature'][self.ear][self.method]
        hrirs_r = self.data[data_idx]['feature']['R' if self.ear == 'L' else 'L'][self.method]
        coord = self.data[data_idx]['feature'][self.ear][self.pos]
        e_l = self.data[data_idx]['label'][self.ear]['feature']
        tbody = self.data[data_idx]['label']['T']['feature']
        e_l = e_l + tbody
        # e_m = data['label'][self.ear]['metric']
        # t_l = data['label']['T']['feature']
        # t_m = data['label']['T']['metric']

        tar_pos = coord[tgts_idx]
        target = np.expand_dims(hrirs[tgts_idx], axis=0)
        target_r = np.expand_dims(hrirs_r[tgts_idx], axis=0)
        tar_pos = np.expand_dims(tar_pos, axis=0)
        inputs = []
        src_pos = []
        srcs_idx = self.nbhd[tgts_idx]
        for si in srcs_idx:
            inputs.append(np.expand_dims(hrirs[si], axis=0))
            src_pos.append(np.expand_dims(coord[si], axis=0))

        inputs = np.concatenate(inputs, axis=0)
        src_pos = np.concatenate(src_pos, axis=0)
        inputs,_ = logmag_phase(torch.tensor(inputs))
        inputs = torch.tensor(np.vstack([inputs, torch.ones_like(torch.randn((1,129)))]), dtype=torch.float)
        # inputs = torch.tensor(np.vstack([inputs, torch.zeros_like(torch.randn((1,129)))]), dtype=torch.float)
        src_pos = np.vstack([src_pos, tar_pos])
        measure = np.expand_dims(np.array(e_l), axis=0)
        measure = torch.tensor(measure, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        tar_pos = torch.tensor(tar_pos, dtype=torch.float)
        num_nodes = len(inputs)

        src_mea_pos = torch.cat((torch.tensor(src_pos),measure.repeat(num_nodes,1)),dim=-1)

        # Calculate edges and edge weights
        edges = []
        edge_weights = []

        # Compute edges and weights between nodes
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                src_pos_ = src_pos[i]
                tgt_pos_ = src_pos[j]
                norm_src = np.linalg.norm(src_pos_)
                norm_tgt = np.linalg.norm(tgt_pos_)
                dot_product = np.dot(src_pos_, tgt_pos_)
                cos_angle = dot_product / (norm_src * norm_tgt)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_degrees = np.degrees(angle)

                if angle_degrees < self.lowrange and angle_degrees > 0:
                    distance = np.linalg.norm(src_pos_ - tgt_pos_)
                    weight = np.exp(-distance ** 2 / (2 * self.sigma ** 2))  # Gaussian kernel
                    # weight = F.mse_loss(inputs[i],inputs[j]).pow(0.5)
                    edges.append([i, j])
                    edges.append([j, i])  # Add reverse edge for undirected graph
                    edge_weights.append(weight)
                    edge_weights.append(weight)
                    # edge_weights.append(np.concatenate([np.array([weight]), src_pos_, tgt_pos_]))
                    # edge_weights.append(np.concatenate([np.array([weight]), tgt_pos_, src_pos_]))

                    # Convert to PyTorch tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 将列表转换为 numpy.ndarray
        edge_weights_array = np.array(edge_weights)
        # 将 numpy.ndarray 转换为 PyTorch 张量
        edge_attr = torch.tensor(edge_weights_array, dtype=torch.float)
        # Convert target and positions to PyTorch tensors
        inputs = inputs.to(dtype=torch.float)
        src_mea_pos = src_mea_pos.to(dtype=torch.float)
        # return inputs, target, measure, src_pos, tar_pos, target_r
        # Create graph data object
        data = Data(x=inputs, edge_index=edge_index, edge_attr=edge_attr)
        data_pos = Data(x=src_mea_pos, edge_index=edge_index, edge_attr=edge_attr)
        return target, tar_pos, data,measure,data_pos

    def calculate_angle(self, p1, p2):
        """计算两个点之间的夹角"""
        dot_product = np.dot(p1, p2)
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        cosine_angle = dot_product / (norm_p1 * norm_p2)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def calculate_virtual_hrtf(self, hrirs, src_pos, target_pos, angle_threshold_inner=6):
        """
        计算目标点的虚拟HRTF，使用angle_threshold_inner角度内的邻居点HRTF均值。

        参数:
        - hrirs: 邻居点的HRTF数据
        - src_pos: 邻居点的坐标
        - target_pos: 目标点的坐标
        - angle_threshold_inner: 内部邻居点之间的角度阈值（度）

        返回:
        - virtual_hrtf: 目标点的虚拟HRTF
        """
        neighbors_hrtf = []
        for i, pos in enumerate(src_pos):
            angle = self.calculate_angle(target_pos, pos)
            if angle <= angle_threshold_inner:
                neighbors_hrtf.append(hrirs[i])
        virtual_hrtf = np.mean(neighbors_hrtf, axis=0)
        return virtual_hrtf

    def build_graph_with_virtual_target(self, hrirs, src_pos, target_pos, angle_threshold_outer=10.0,
                                        angle_threshold_inner=6):
        """
        构建图，包含目标点，节点为每个点的HRTF，边为角度小于5度的点。

        参数:
        - hrirs: 邻居点的HRTF数据
        - src_pos: 邻居点的坐标
        - target_pos: 目标点的坐标
        - angle_threshold_outer: 外部邻居点之间的角度阈值（度）
        - angle_threshold_inner: 内部邻居点之间的角度阈值（度）

        返回:
        - data: PyTorch Geometric的图数据对象
        """
        # 提取外部邻居点
        outer_neighbors_hrirs = []
        outer_neighbors_pos = []
        for i, pos in enumerate(src_pos):
            angle = self.calculate_angle(target_pos, pos)
            if angle <= angle_threshold_outer:
                outer_neighbors_hrirs.append(hrirs[i])
                outer_neighbors_pos.append(pos)

        # 计算目标点的虚拟HRTF
        virtual_hrtf = self.calculate_virtual_hrtf(outer_neighbors_hrirs, outer_neighbors_pos, target_pos,
                                                   angle_threshold_inner)

        # 包含目标点在内的所有点的特征
        all_hrirs = [virtual_hrtf] + outer_neighbors_hrirs
        all_pos = [target_pos] + outer_neighbors_pos

        num_nodes = len(all_hrirs)
        x = torch.tensor(all_hrirs, dtype=torch.float)

        # 初始化边索引
        edge_index = []

        # 构建边索引
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                angle = self.calculate_angle(all_pos[i], all_pos[j])
                if angle <= angle_threshold_inner:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index)

        return data

    def prep_nbhd(self):
        subj_idx = 0
        data_idx = self.files[subj_idx]
        data_tgt = self.data_path[data_idx]
        data = load_dict(data_tgt)
        self.nbhd = {}
        print("[Loader] Gathering neighborhood info")
        for srcs_idx in range(self.num_grid):
            self.nbhd[srcs_idx] = []
        for srcs_idx in tqdm(range(self.num_grid)):
            coord = data['feature'][self.ear][self.pos]
            src_pos = coord[srcs_idx]
            for tgts_idx in range(srcs_idx, self.num_grid):
                tar_pos = coord[tgts_idx]
                dist = np.sqrt(np.sum((src_pos - tar_pos) ** 2))
                if dist < self.range and dist > 0:
                    if not (tgts_idx in self.nbhd[srcs_idx]):
                        self.nbhd[srcs_idx].append(tgts_idx)
                    if not (srcs_idx in self.nbhd[tgts_idx]):
                        self.nbhd[tgts_idx].append(srcs_idx)

    def prep_nbhdnew(self):
        subj_idx = 0
        data_idx = self.files[subj_idx]
        data_tgt = self.data_path[data_idx]
        data = load_dict(data_tgt)
        self.nbhd = {}
        print("[Loader] Gathering new neighborhood info")
        for srcs_idx in range(self.num_grid):
            self.nbhd[srcs_idx] = []
        # for srcs_idx in tqdm(range(self.num_grid)):
        for srcs_idx in tqdm(range(self.num_grid)):
            coord = data['feature'][self.ear][self.pos]
            src_pos = coord[srcs_idx]
            for tgts_idx in range(srcs_idx, self.num_grid):
                tar_pos = coord[tgts_idx]
                # 计算目标点的模长（假设目标点是单位向量）
                norm_target = np.linalg.norm(tar_pos)
                # 计算每个点的模长
                norm_points = np.linalg.norm(src_pos)
                # 计算每个点与目标点的点积
                dot_products = np.dot(src_pos, tar_pos)
                # 计算余弦值
                cos_angles = dot_products / (norm_points * norm_target)
                # 计算夹角（弧度）
                angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
                # 转换为度数
                angles_degrees = np.degrees(angles)

                # dist = np.sqrt(np.sum((src_pos - tar_pos)**2))
                if angles_degrees < self.range and angles_degrees > 0:
                    if not (tgts_idx in self.nbhd[srcs_idx]):
                        self.nbhd[srcs_idx].append(tgts_idx)
                    if not (srcs_idx in self.nbhd[tgts_idx]):
                        self.nbhd[tgts_idx].append(srcs_idx)


class Trainset(GenericDataset):

    def __init__(
            self, args,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2,
            n_samples=1,
            sort_by_dist=False,
    ):
        super().__init__(
            args, directory, subj_list,
            method=method, ear=ear, patch_range=patch_range,
            n_samples=n_samples, sort_by_dist=sort_by_dist,
        )
        print(f"[Loader] Train subject IDs:\t {self.files}")


class Testset(GenericDataset):

    def __init__(
            self, args,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2, mode='Test',
            n_samples=1,
            sort_by_dist=False,
    ):
        super().__init__(
            args, directory, subj_list,
            method=method, ear=ear, patch_range=patch_range,
            n_samples=n_samples, sort_by_dist=sort_by_dist,
        )
        print(f"[Loader] {mode} subject IDs:\t {self.files}")

class CoarseSet(torch.utils.data.Dataset):

    def __init__(
            self,
            name,
            directory=None,
            subj_list=None,
            method='S', ear='L',
            patch_range=0.2,
            num_grid = 1730,
            n_samples=1,
            sort_by_dist=False,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None,
            scale_factor=1,
        ):

        self.num_grid=num_grid
        self.method = method     # method \in ['S', 'M']
        self.ear = ear           # ear \in ['L', 'R']
        self.num_grid = num_grid
        if directory is not None:
            self.data_path = sorted(glob.glob(os.path.join(directory,'*.pkl')))
        if subj_list is not None:
             self.files = subj_list
        self.scale_factor=scale_factor
        self.xc = x_constraint
        self.yc = y_constraint
        self.zc = z_constraint
        self.src_sel_grid = 0
        self.tar_sel_grid = 0
        self.src_selected = []
        self.tar_selected = []
        self.range = patch_range
        self.n_samples = n_samples
        self.sort_by_dist = sort_by_dist

        self.pos = 's_pos' if method=='S' else 'm_pos'
        if directory is not None:
            _data = load_dict(self.data_path[0])
            self.coord = _data['feature'][self.ear][self.pos]
            self.select_index()
            self.prep_nbhd()
            mins = np.inf
            maxs = 0
            ids = self.nbhd.keys()
            for i in ids:
                lens = len(self.nbhd[i])
                if lens < mins:
                    mins = lens
                if lens > maxs:
                    maxs = lens
            self.reorder_index()
            print(f"*** number of nbhd points in [{mins}, {maxs}]")

    def __len__(self):
        return self.tar_sel_grid * len(self.files)

    def __getitem__(self, index):

        subj_idx = int(index // self.tar_sel_grid)
        tgts_idx = self.tar_selected[int(index % self.tar_sel_grid)]
        data_idx = self.files[subj_idx]
        data_tgt = self.data_path[data_idx]
        data = load_dict(data_tgt)
        hrirs = data['feature'][self.ear][self.method]
        e_l = data['label'][self.ear]['feature']
        tar_pos = self.coord[tgts_idx]

        # load srcs
        inputs = []
        src_pos = []
        if self.sort_by_dist:
            srcs_idx = sorted_choices(p=tgts_idx, qs=self.nbhd[tgts_idx], k=self.n_samples, p_sys=self.coord)
        else:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        d = self.n_samples - len(srcs_idx)
        if d > 0:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        for si in srcs_idx:
            inputs.append(np.expand_dims(hrirs[si], axis=0))
            src_pos.append(np.expand_dims(self.coord[si], axis=0))

        if tgts_idx == 0:
            linear = hrirs[1]
        elif tgts_idx == self.num_grid-1:
            linear = hrirs[-1]
        else:
            linear = 0.5 * (hrirs[tgts_idx-1] + hrirs[tgts_idx+1])

        inputs = np.concatenate(inputs, axis=0)
        target = np.expand_dims(hrirs[tgts_idx], axis=0)
        linear = np.expand_dims(linear, axis=0)
        src_pos = np.concatenate(src_pos, axis=1)
        tar_pos = np.expand_dims(tar_pos, axis=0)
        measure = np.expand_dims(np.array(e_l), axis=0)
        return inputs, target, linear, measure, src_pos, tar_pos

    def select_index(self):
        criterion = []
        for i in range(self.num_grid):
            x, y, z = self.coord[i]
            ang = self.coord_to_ang(x,y,z)
            if not(ang in criterion):
                criterion.append(ang)
        criterion = criterion[0::self.scale_factor]
        def scale_selected(x,y,z):
            cond = False
            ang = self.coord_to_ang(x,y,z)
            if ang in criterion:
                cond = True
            return cond
        def satisfied_constraint(x,y,z,xc,yc,zc):
            cond = True
            if (xc is not None) and round(x,5)!=xc:
                cond = False
            if (yc is not None) and round(y,5)!=yc:
                cond = False
            if (zc is not None) and round(z,5)!=zc:
                cond = False
            return cond
        for i in range(self.num_grid):
            x, y, z = self.coord[i]
            if scale_selected(x,y,z):
                self.src_selected.append(i)
            if satisfied_constraint(x,y,z,self.xc,self.yc,self.zc):
                self.tar_selected.append(i)
        self.src_sel_grid = len(self.src_selected)
        self.tar_sel_grid = len(self.tar_selected)

    def prep_nbhd(self):
        self.nbhd = {}     # nbhd among coarse grid
        print("[Loader] Gathering neighborhood info")
        for tgts_idx in range(self.tar_sel_grid):
            self.nbhd[self.tar_selected[tgts_idx]] = []
        for tgts_idx in tqdm(range(self.tar_sel_grid)):
            tar_pos = self.coord[self.tar_selected[tgts_idx]]
            for srcs_idx in range(self.src_sel_grid):
                tgts_idx_g = self.tar_selected[tgts_idx]
                srcs_idx_g = self.src_selected[srcs_idx]
                src_pos = self.coord[srcs_idx_g]
                dist = np.sqrt(np.sum((src_pos - tar_pos)**2))
                if dist < self.range:
                    if not (srcs_idx_g in self.nbhd[tgts_idx_g]):
                        self.nbhd[tgts_idx_g].append(srcs_idx_g)

    def coord_to_ang(self,x,y,z):
        r = np.sqrt(x**2 + y**2)
        az = - np.arctan2(x,y) + np.pi/2
        el = np.arctan2(z,r)
        az += 2*np.pi if az < 0 else 0
        if (self.yc is not None) and (x < 0):
            el = np.pi - el
        if (self.xc is not None) and (y < 0):
            el = np.pi - el
        el += 2*np.pi if el < 0 else 0
        ang = az if self.zc is not None else el
        return ang

    def reorder_index(self):
        d = {}
        for i in range(self.src_sel_grid):
            x, y, z = self.coord[self.src_selected[i]]
            ang = self.coord_to_ang(x,y,z)
            if ang in d.keys():
                ang += 1e-2 * np.random.random()
            d[ang] = self.src_selected[i]
        od = dict(sorted(d.items()))
        i = 0
        for key, val in od.items():
            self.src_selected[i] = val
            i += 1
        d = {}
        for i in range(self.tar_sel_grid):
            x, y, z = self.coord[self.tar_selected[i]]
            ang = self.coord_to_ang(x,y,z)
            if ang in d.keys():
                ang += 1e-2 * np.random.random()
            d[ang] = self.tar_selected[i]
        od = dict(sorted(d.items()))
        i = 0
        for key, val in od.items():
            self.tar_selected[i] = val
            i += 1
