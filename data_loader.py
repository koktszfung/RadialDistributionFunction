import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from pattern import Pattern


class DatasetBandStructureToPlaneGroup(Dataset):
    def __init__(self, device, in_dirs):
        self.data_inputs = []
        self.data_labels = []
        for in_dir in in_dirs:
            for root, dir_names, file_names in os.walk(in_dir):
                for i, file_name in enumerate(file_names):
                    data_label_np = np.array([Pattern.group_numbers[file_name.split(".")[0]]])
                    data_input_nps = np.random.permutation(list(np.load(os.path.join(root, file_name)).values())[0])
                    for data_input_np in data_input_nps:
                        self.data_inputs.append(torch.from_numpy(data_input_np.flatten().T).float().to(device))
                        self.data_labels.append(torch.from_numpy(data_label_np).long().to(device))
                    print(f"\r\tload files: {i}/{len(file_names)}", end="")
                print(f"\rload files: {len(file_names)}")
        self.len = len(self.data_inputs)
        print(f"number of data: {len(self)}")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_inputs[index], self.data_labels[index]


def get_valid_train_loader(dataset, batch_size, valid_size):
    num_train = len(dataset)
    indices = range(num_train)
    split = int(valid_size * num_train)

    valid_sampler = SubsetRandomSampler(indices[:split])
    train_sampler = SubsetRandomSampler(indices[split:])

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    return valid_loader, train_loader
