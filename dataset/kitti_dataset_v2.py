import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

from torch_geometric.data import Data
import torch_geometric.transforms as T



def load_files(folder):
    """Load all files in a folder and sort."""
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder))
        for f in fn
    ]
    file_paths.sort()
    return file_paths

def normalize_pc(data):
    """
    normalize data to [-1, 1], 0 mean
    """
    centroid = np.mean(data, axis=-2, keepdims=True, dtype=np.float32)
    data = data - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(data ** 2, axis=-1)))

    data = data /furthest_distance
    return data

class KittiDataset_2(Dataset):
    '''
    Multi sequence training on Kitti dataset
    Parameter:
        root: dir of kitti dataset (sequence/)
        npoints: number of random sampled points from raw points
        input_num: input point cloud number
        pred_num: predicted point cloud number
        seqs: sequence list
    '''
    def __init__(self, root, npoints, input_num, pred_num, seqs):
        super(KittiDataset_2, self).__init__()

        self.root = root
        self.seqs = seqs
        self.input_num = input_num
        self.pred_num = pred_num
        self.npoints = npoints
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        dataset = []
        for seq in self.seqs:
            dataroot = os.path.join(self.root, seq, 'velodyne')
            datapath = glob.glob(os.path.join(dataroot, '*.bin'))
            datapath = sorted(datapath)
            max_ind = len(datapath)
            ini_index = 0
            interval = self.input_num + self.pred_num
            while (ini_index < max_ind - interval):
                paths = []
                for i in range(interval):
                    # curr_path = os.path.join(seq, 'velodyne',datapath[ini_index+i])
                    curr_path = datapath[ini_index+i]

                    paths.append(curr_path)
                ini_index += interval
                dataset.append(paths)
        return dataset
    
    def get_cloud(self, filename):
        pc = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
        N = pc.shape[0]
        if N >= self.npoints:
            sample_idx = np.random.choice(N, self.npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(N), np.random.choice(N, self.npoints-N, replace=True)), axis=-1)
        xyz = pc[sample_idx, :3].astype('float32')
        # fea = pc[sample_idx, 3:].astype('float32')

        # xyz = normalize_pc(xyz)
        xyz = torch.from_numpy(xyz).t()
        # fea = torch.from_numpy(fea).t()
        return xyz

    def __getitem__(self, index):
        paths = self.dataset[index]
        input_pc_list = []
        # input_fea_list = []
        for i in range(self.input_num):
            input_pc_name = paths[i]
            # input_pc = self.get_cloud(os.path.join(self.root, input_pc_name))
            input_pc = self.get_cloud(input_pc_name)
            input_pc_list.append(input_pc)
            # input_fea_list.append(input_fea)

        # input_pc = torch.stack(input_pc_list, dim=0)
        
        output_pc_list = []
        for i in range(self.input_num, self.input_num+self.pred_num):
            output_pc_name  = paths[i]
            # output_pc = self.get_cloud(os.path.join(self.root, output_pc_name))
            output_pc= self.get_cloud(output_pc_name)
            output_pc_list.append(output_pc)
        # output_pc = torch.stack(output_pc_list, dim=0)
        
        return input_pc_list, output_pc_list


    
    def __len__(self):
        return len(self.dataset)
    

if __name__ == "main":

    datapath = "../kittiraw/dataset/sequences"
    # train_seqs = ['00','01','02','03','04','05']
    pc_seq_dataset = KittiDataset_2(root=datapath, npoints=4096, input_num=5, pred_num=5, seqs=['147'])
    
    input_pc, gt_out_pc = pc_seq_dataset[147]
    print(input_pc.shape)
    print(gt_out_pc.shape)