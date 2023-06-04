import torch
from torch import nn

from train_model import load_model_and_create_optimizer
from wideSegFormer import SegformerDecodeHead
from smallModel import SmallSegFormer

from transformers import SegformerForSemanticSegmentation

class TwinSegFormer(nn.Module):
    def __init__(self, device="cpu"):
        super(TwinSegFormer, self).__init__()
        self.handSegFormer, _ = load_model_and_create_optimizer(
            model_class = SmallSegFormer,
            model_params = {
                "freeze_encoder": True, 
                "segFormerNumber": 2
            },
            model_name = "handModel3",
            device = device,
            epoch = 38,
            lr=1e-5
        )
        self.handSegFormer = self.handSegFormer.segFormer.segformer
        
        self.bodySegFormer, _ = load_model_and_create_optimizer(
                model_class = SmallSegFormer,
                model_params = {
                    "freeze_encoder": False, 
                    "segFormerNumber": 2
                },
                model_name = "bodyModel",
                device = device,
                epoch = 46,
                lr=1e-5
        )
        self.bodySegFormer = self.bodySegFormer.segFormer.segformer
        self.decode_head = SegformerDecodeHead(1).to(device)
    
    def forward(self, x):
        x = x.float() / 255
        emb_hand = self.handSegFormer(x, output_hidden_states=True)
        emb_body = self.bodySegFormer(x, output_hidden_states=True)
        
        emb = []
        for i in range(4):
            emb.append(torch.concat([emb_hand.hidden_states[i], emb_body.hidden_states[i]], dim=1))

        output = self.decode_head(emb)
        return output