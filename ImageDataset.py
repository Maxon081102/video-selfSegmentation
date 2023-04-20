import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from imgPrepare import img_prepare

class ImageDataset(Dataset):
    def __init__(self, img_names, img_prepare, transform):
        self.data = []
        for img_name in tqdm(img_names):
            img, mask = img_prepare(img_name, transform)
            self.data.append((img, mask))

    def __getitem__(self, index):
        img, mask = self.data[index]
        return img, mask

    def __len__(self):
        return len(self.data)

def get_img_for_train_and_val(data_folder, p=0.7, seed=777, img_count=100000):
    train = []
    test = []
    np.random.seed(seed)
    for img_name in os.listdir(data_folder + "/X"):
        if os.path.isfile(data_folder + "/Y/" + img_name.replace("X", "Y")):
            if np.random.rand() >= p:
                if len(test) < img_count:
                    test.append(img_name)
            else:
                if len(train) < img_count:
                    train.append(img_name)
    return train, test

def collate_fn(batch):
    img_batch = torch.stack([elem[0] for elem in batch])
    mask_batch = torch.stack([elem[1] for elem in batch])
    return img_batch, mask_batch