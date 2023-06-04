import torch

from torch import nn

from transformers import SegformerForSemanticSegmentation

class SmallSegFormer(nn.Module):
    def __init__(self, freeze_encoder=True, segFormerNumber=0):
        super(SmallSegFormer, self).__init__()
        if segFormerNumber == 0:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        elif segFormerNumber == 1:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        elif segFormerNumber == 2:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        elif segFormerNumber == 3:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        elif segFormerNumber == 4:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        elif segFormerNumber == 5:
            self.segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

        if freeze_encoder:
            for param in self.segFormer.parameters():
                param.requires_grad = False
    
        if segFormerNumber <= 1:
            self.segFormer.decode_head.classifier = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.segFormer.decode_head.classifier = nn.Conv2d(768, 1, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        x = x.float()
        output = self.segFormer(x)
        return output.logits


class Sherlock(nn.Module):
    def __init__(self, segFormerNumber=0, mask_size=128):
        super(Sherlock, self).__init__()
        self.segFormer = SmallSegFormer(False, segFormerNumber)
        self.W1 = nn.Linear(mask_size, 1)
        self.W2 = nn.Linear(mask_size, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.segFormer(x).permute(0, 2, 3, 1)
        x = x.squeeze(-1)
        output = self.W1(x)
        output = self.W2(output.transpose(1, 2))
        output = output.view(-1)
        return output

