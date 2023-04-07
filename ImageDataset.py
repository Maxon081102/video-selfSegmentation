import os
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from imgPrepare import img_prepare
class ImageDataset(Dataset):
    def __init__(self, data_folder, img_prepare, transform):
        self.data = []
        for img_name in tqdm(os.listdir(data_folder + "/X")):
            if os.path.isfile(data_folder + "/Y/" + img_name.replace("X", "Y")):
                img, mask = img_prepare(img_name, transform)
                self.data.append((img, mask))

    def __getitem__(self, index):
        img, mask = self.data[index]
        return img, mask

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    img_batch = torch.stack([elem[0] for elem in batch])
    mask_batch = torch.stack([elem[1] for elem in batch])
    return img_batch, mask_batch