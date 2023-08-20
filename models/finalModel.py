import torch
# import numpy as np

from torch import nn
from torchvision import transforms as tf
from train.train_model import load_model_and_create_optimizer

from models.smallModel import Sherlock
from models.wideSegFormer import WideSegFormer
from models.TwinSegFormer import TwinSegFormer

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        # self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        # self.img_prepare = image_prepare = tf.Compose([
        #     tf.ToPILImage(),
        #     tf.ToTensor(),
        #     tf.Resize(size = (512,512))
        # ])
        self.img_prepare = tf.Resize(size = (512,512))
        # self.img_prepare = torch.jit.script(self.img_prepare)
        self.wideSegFormer, _ = load_model_and_create_optimizer(
            model_class = WideSegFormer,
            model_params = {
                "out_size": 1, 
                "frozen_encoder": False, 
                "without_annotation": True
            },
            model_name = "test2",
            device = self.device,
            epoch = 42,
            lr=1e-5,
            
        )

        self.sherlock, _ = load_model_and_create_optimizer(
            model_class = Sherlock,
            model_params = {"segFormerNumber": 1},
            model_name = "sherlock1",
            device = self.device,
            epoch = 30,
            lr=1e-5
        )

        self.wideSegFormer.eval()
        self.sherlock.eval()

        # self.wideSegFormer = torch.jit.trace(self.wideSegFormer, torch.rand((1,3,512,512),
        #                               dtype=torch.float).to(self.device))

        # self.sherlock = torch.jit.trace(self.sherlock, torch.rand((1,3,512,512),
        #                               dtype=torch.float).to(self.device))

    def forward(self, x):
        img_size = (x.shape[1], x.shape[2])
        x = self.img_prepare(x)
        x = x.to(self.device)
        # with torch.no_grad():
        #     there_are_people = self.sherlock(x[None])[0]
        there_are_people = self.sherlock(x[None])[0]
        if there_are_people < 0.5:
            return torch.zeros(img_size)
        
        # with torch.no_grad():
        #     res = self.wideSegFormer(x[None])[0]
        res = self.wideSegFormer(x[None])[0]
        res = nn.functional.sigmoid(res.permute(1, 2, 0)).squeeze(-1)
        res = (res >= 0.5).float()
        res = nn.functional.interpolate(
            res[None, None], size=(img_size[0], img_size[1]), mode="bilinear", align_corners=False
        )[0][0]
        return res.cpu()

class FinalModel2(nn.Module):
    def __init__(self):
        super(FinalModel2, self).__init__()
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.img_prepare = tf.Resize(size = (512,512))
        # self.img_prepare = torch.jit.script(self.img_prepare)
        self.twinSegFormer, _ = load_model_and_create_optimizer(
            model_class = TwinSegFormer,
            model_params = {},
            model_name = "finalModel2",
            chkp_folder = "models",
            device = self.device,
            epoch = 25,
            lr=1e-5,
            model_dir = False,
        )

        self.sherlock, _ = load_model_and_create_optimizer(
            model_class = Sherlock,
            model_params = {"segFormerNumber": 1},
            model_name = "sherlock1",
            chkp_folder = "models",
            device = self.device,
            epoch = 10,
            lr=1e-5,
            model_dir = False,
        )

        self.twinSegFormer.eval()
        self.sherlock.eval()

        # self.twinSegFormer = torch.jit.trace(self.twinSegFormer, torch.rand((1,3,512,512),
        #                               dtype=torch.float).to(self.device))

        # self.sherlock = torch.jit.trace(self.sherlock, torch.rand((1,3,512,512),
        #                               dtype=torch.float).to(self.device))

    def forward(self, x):
        img_size = (x.shape[1], x.shape[2])
        x = self.img_prepare(x)
        x = x.to(self.device)
        # with torch.no_grad():
        #     there_are_people = self.sherlock(x[None])[0]
        there_are_people = self.sherlock(x[None])[0]
        if there_are_people < 0.5:
            return torch.zeros(img_size)
        
        # with torch.no_grad():
        #     res = self.wideSegFormer(x[None])[0]
        res = self.twinSegFormer(x[None])[0]
        res = nn.functional.sigmoid(res.permute(1, 2, 0)).squeeze(-1)
        res = (res >= 0.5).float()
        res = nn.functional.interpolate(
            res[None, None], size=(img_size[0], img_size[1]), mode="bilinear", align_corners=False
        )[0][0]
        return res.cpu()


# class ToTensor(object):
#     def __call__(self, sample):
#         return torch.from_numpy(sample).short().permute((2, 0, 1))