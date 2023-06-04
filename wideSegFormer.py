import copy
import torch
from torch import nn

from transformers import SegformerForSemanticSegmentation

class SegformerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(nn.Module):
    def __init__(self, out_size=1):
        super().__init__()

        mlps = []
        decoder_hidden_size = 768 * 2
        hidden_sizes = [128, 256, 640, 1024]
        num_encoder_blocks = len(hidden_sizes)
        for i in range(num_encoder_blocks):
            mlp = SegformerMLP(input_dim=hidden_sizes[i], output_dim=decoder_hidden_size)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU()
        self.out_size = out_size

        classifier_dropout_prob = 0.1
        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Conv2d(decoder_hidden_size, out_size, kernel_size=1)

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)

            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        logits = self.classifier(hidden_states)

        return logits

class WideSegFormer(nn.Module):
    def __init__(self, out_size=1, frozen_encoder=True, without_annotation=False):
        super(WideSegFormer, self).__init__()
        segFormer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        self.without_annotation = without_annotation
        self.segFormer_for_annotate =segFormer.segformer
        if frozen_encoder:
            for param in self.segFormer_for_annotate.parameters():
                param.requires_grad = False
        # if without_annotation:
            # self.segFormer_for_img = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").segformer
            # self.segFormer_for_img = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").segformer
        self.segFormer_for_img = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").segformer
        self.decode_head = SegformerDecodeHead(out_size)
    
    def forward(self, x, x_annotate=None):
        x = x.float() / 255
        if self.without_annotation:
            x_annotate = x.clone()
            emb_annotate = self.segFormer_for_annotate(x_annotate, output_hidden_states=True)
            emb_img = self.segFormer_for_img(x, output_hidden_states=True)
        else:
            # TODO: пока не gray
            x_annotate = x_annotate.float() / 255
            emb_annotate = self.segFormer_for_annotate(x_annotate, output_hidden_states=True)
            emb_img = self.segFormer_for_img(x, output_hidden_states=True)
            # emb_img = self.segFormer_for_annotate(x, output_hidden_states=True)
        
        emb = []
        for i in range(4):
            emb.append(torch.concat([emb_annotate.hidden_states[i], emb_img.hidden_states[i]], dim=1))

        output = self.decode_head(emb)
        return output