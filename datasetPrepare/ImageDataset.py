import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
# from tqdm.notebook import tqdm
from datasetPrepare.imgPrepare import img_prepare

class ImageDataset(Dataset):
    def __init__(self, img_names, img_prepare, transform, annotate=False, use_filter=False, only_img=False):
        self.data = img_names
        self.img_prepare = img_prepare
        self.transform = transform
        self.annotate = annotate
        self.use_filter = use_filter
        self.only_img = only_img

    def __getitem__(self, index):
        return self.img_prepare(self.data[index], self.transform, self.annotate, self.use_filter, self.only_img)

    def __len__(self):
        return len(self.data)

def get_img_for_train_and_val(data_folder, p=0.9, seed=777, img_count=100000):
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
    if type(batch[0][1]) == bool:
        mask_batch = torch.Tensor([elem[1] for elem in batch])
        return img_batch, mask_batch

    mask_batch = torch.stack([elem[1] for elem in batch])

    if len(batch[-1]) == 3:
        annotate_batch = torch.stack([elem[2] for elem in batch])
        return img_batch, mask_batch, annotate_batch
    return img_batch, mask_batch