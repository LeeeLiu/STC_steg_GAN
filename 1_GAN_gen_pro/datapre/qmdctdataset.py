import os

import numpy as np

import torch
from torch.utils.data import Dataset


def read_qmdct(filepath):
    with open(filepath, 'r') as f:
        total = []
        for line in f.readlines():
            line_list = line.split()
            tmp = []
            for coe in line_list:
                tmp.append(float(coe))
            total.append(tmp)

        qmdct_array = np.asarray(total).astype('float32')
        # print(qmdct_array.shape)

        return qmdct_array


class QmdctDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, item):
        qmdct_name = os.listdir(self.root_dir)[item]    # 这里 想记录cover名字 好与stego对应
        qmdct_path = os.path.join(self.root_dir, qmdct_name)
        qm_array = read_qmdct(qmdct_path)
        if self.transform:
            qm_array = self.transform(qm_array)
        qm_tensor_norm = torch.Tensor(qm_array).unsqueeze(0)

        return qm_tensor_norm


def main():
    read_qmdct("demo.txt")


if __name__ == "__main__":
    main()