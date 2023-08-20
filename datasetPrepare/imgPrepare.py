import os
import cv2
import torch
import numpy as np

from torchvision import transforms as tf

def img_prepare(img_name, transform, annotate=False, use_filter=False, only_img=False):
    # TODO : add normalization
    img = cv2.imread(os.path.join("data", 'X', img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mask = get_img_mask(img_name)

    if annotate and not use_filter:
        annotate_name = img_name.replace("X", "Z")
        img_annotate = cv2.imread(os.path.join("data", 'Z', annotate_name))
        # img_annotate = cv2.cvtColor(img_annotate, cv2.COLOR_BGR2GRAY)
        img_annotate = cv2.cvtColor(img_annotate, cv2.COLOR_BGR2RGB)
        output = transform({'image': img, 'mask': img_mask, 'annotate': img_annotate})
        return output["image"], output["mask"], output["annotate"]
    
    if use_filter and annotate:
        annotate = sobel_edge_detection(img)
        output = transform({'image': img, 'mask': img_mask, 'annotate': annotate})
        return output["image"], output["mask"], output["annotate"]

    output = transform({'image': img, 'mask': img_mask})
    if only_img:
        return output["image"], (output["mask"].sum() > 0).item()
    return output["image"], output["mask"]

# def img_prepare(img_name, transform):
#     # TODO : add normalization
#     img = cv2.imread(os.path.join("data", 'X', img_name))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_mask = get_img_mask(img_name)

#     if annotate and not use_filter:
#         annotate_name = img_name.replace("X", "Z")
#         img_annotate = cv2.imread(os.path.join("data", 'Z', annotate_name))
#         # img_annotate = cv2.cvtColor(img_annotate, cv2.COLOR_BGR2GRAY)
#         img_annotate = cv2.cvtColor(img_annotate, cv2.COLOR_BGR2RGB)
#         output = transform({'image': img, 'mask': img_mask, 'annotate': img_annotate})
#         return output["image"], output["mask"], output["annotate"]
    
#     if use_filter and annotate:
#         annotate = sobel_edge_detection(img)
#         output = transform({'image': img, 'mask': img_mask, 'annotate': annotate})
#         return output["image"], output["mask"], output["annotate"]

#     output = transform({'image': img, 'mask': img_mask})
#     if only_img:
#         return output["image"], (output["mask"].sum() > 0).item()
#     return output["image"], output["mask"]

def sobel_edge_detection(img):
    a = 0
    b = 1
    c = 2
    f_x = np.array([
        [-b, -c, -b],
        [a, a, a],
        [b, c, b]
    ])
    f_y = np.array([
        [-b, a, b],
        [-c, a, c],
        [-b, a, b]
    ])
    
    f_z = np.array([
        [a, -b, -c],
        [b, a, -b],
        [c, b, a]
    ])
    f_k = np.array([
        [-c, -b, a],
        [-b, a, b],
        [a, b, c]
    ])

    mask1 = make_filter_mask(f_x, img)
    mask2 = make_filter_mask(f_z, img)
    mask3 = make_filter_mask(f_y, img)
    mask4 = make_filter_mask(f_k, img)
    mask5 = make_filter_mask(-f_x, img)
    mask6 = make_filter_mask(-f_z, img)
    mask7 = make_filter_mask(-f_y, img)
    mask8 = make_filter_mask(-f_k, img)
    mask = ((mask1 + mask2 + mask3 + mask4 + mask5 + mask6 + mask7 + mask8)).astype(bool).astype(int) * 255 
    return mask

def make_filter_mask(filter_, img):
    res = cv2.filter2D(img,-1,filter_)
    mask = (res.sum(axis=2) > 50)
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask = np.repeat(mask,3, axis=2).astype(int)
    return mask

def create_img_transform(size, val=False, annotate=False):
    return tf.Compose([
        ToTensor(annotate),
        ResizeOrRandomCrop(size, val, annotate),
        Rotate(val, annotate)
    ])

def get_img_mask(img_name):
    img_name = img_name.replace("X", "Y")
    img = cv2.imread(os.path.join("data", 'Y', img_name))
    assert img is not None, "img doesn't exist"
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_torch = torch.Tensor(img).long()
    mask = (img - torch.argmax(torch.bincount(img_torch.view(-1))).item()).sum(axis=2).astype(bool)
    return torch.Tensor(mask).bool()

class ToTensor(object):
    def __init__(self, annotate):
        self.annotate = annotate

    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))
        if self.annotate:
            annotate = sample["annotate"]
            annotate = annotate.transpose((2, 0, 1))
            return {
                'image': torch.from_numpy(image).short(),
                'annotate': torch.from_numpy(annotate).short(),
                'mask': sample['mask']
            }

        return {'image': torch.from_numpy(image).short(),
                'mask': sample['mask']}

class ResizeOrRandomCrop(object):
    def __init__(self, output_size, val, annotate):
        assert isinstance(output_size, (int, tuple))
        self.val = val
        self.annotate = annotate
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.resize = tf.Resize(self.output_size)
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'] 
        if self.annotate:
            annotate = sample['annotate']
        resize = np.random.randint(0, 10) <= 6
        if resize or self.val:
            image = self.resize(image)
            mask = self.resize(mask[None])[0]
            if self.annotate:
                annotate = self.resize(annotate)
        else:
            h, w = image.shape[1:3]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:, top: top + new_h,
                      left: left + new_w]

            mask = mask[top: top + new_h,
                      left: left + new_w]
            
            if self.annotate:
                annotate = annotate[:, top: top + new_h,
                            left: left + new_w]

        if self.annotate:
            return {'image': image, 'mask': mask, 'annotate': annotate}

        return {'image': image, 'mask': mask}

class Rotate(object):
    def __init__(self, val, annotate):
        self.val = val
        self.annotate = annotate

    def __call__(self, sample):
        if self.val:
            return sample
        image, mask = sample['image'], sample['mask']
        if self.annotate:
            annotate = sample['annotate']
        k = np.random.randint(0, 4)
        image = torch.rot90(image, k=k, dims=[1, 2])
        mask = torch.rot90(mask, k=k, dims=[0, 1])

        if self.annotate:
            annotate = torch.rot90(annotate, k=k, dims=[1, 2])
            return {'image': image, 'mask': mask, 'annotate': annotate}

        return {'image': image, 'mask': mask}
