import os
import cv2
import torch
import numpy as np

from torchvision import transforms as tf

def img_prepare(img_name, transform):
    # TODO : add normalization
    img = cv2.imread(os.path.join("data", 'X', img_name))
    img_mask = get_img_mask(img_name)
    output = transform({'image': img, 'mask': img_mask})
    return output["image"], output["mask"]

def create_img_transform(size):
    return tf.Compose([
        ToTensor(),
        ResizeOrRandomCrop(size),
        Rotate()
    ])

def get_img_mask(img_name):
    img_name = img_name.replace("X", "Y")
    img = cv2.imread(os.path.join("data", 'Y', img_name))
    assert img is not None, "img doesn't exist"
    mask = (img - 255).sum(axis=2).astype(bool)
    return torch.Tensor(mask).bool()

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).short(),
                'mask': sample['mask']}

class ResizeOrRandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.resize = tf.Resize(self.output_size)
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'] 
        resize = np.random.randint(0, 10) <= 6
        if resize:
            image = self.resize(image)
            mask = self.resize(mask[None])[0]
        else:
            h, w = image.shape[1:3]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:, top: top + new_h,
                      left: left + new_w]

            mask = mask[top: top + new_h,
                      left: left + new_w]
            
        return {'image': image, 'mask': mask}

class Rotate(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        k = np.random.randint(0, 4)
        image = torch.rot90(image, k=k, dims=[1, 2])
        mask = torch.rot90(mask, k=k, dims=[0, 1])
        return {'image': image, 'mask': mask}
