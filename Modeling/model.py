import torch.nn as nn
from Modeling.Backbone.FocalNet import FocalNet
from Modeling.modules.utils import Hook
from Modeling.modules.attention import (SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP)
from Modeling.modules.pos_embed import (PositionEmbeddingSine)
from Modeling.modules.unet import (
    UnetBlockWide,
    CustomPixelShuffle_ICNR,
    NormType,
    custom_conv_layer
)
import torch
import logging

from typing import List, Dict, Tuple

class DDStainer(nn.Module):
    def __init__(self, 
                encoder_name='focalnet_large',
                pretrained:bool=True,
                num_input_channels:int=3,
                input_size:Tuple[int, int]=(256, 256),
                nf:int=512,
                num_output_channels:int=3,
                last_norm:str='Weight',
                num_queries:int=256,
                num_scales:int=3,
                dec_layers:int=9,
                ) -> None:
        super().__init__()

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.num_input_channels = num_input_channels
        self.input_size = input_size
        self.nf = nf
        self.num_output_channels = num_output_channels
        self.last_norm = getattr(NormType, last_norm)
        self.num_queries = num_queries
        self.num_scales = num_scales
        self.dec_layers = dec_layers

        self.encoder = Encoder(encoder_name, pretrained)
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        self.encoder(test_input)
        self.decoder = Decoder(self.encoder.hooks, 
                                nf=nf, blur=True, 
                                last_norm='Weight', 
                                num_queries=num_queries, 
                                num_scales=num_scales, 
                                dec_layers=dec_layers
                                )

        self.refine_net = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral))


    def forward(self, x:torch.Tensor, do_normalize:bool=True)-> torch.Tensor:

        self.encoder(x, do_normalize)
        output_feat = self.decoder()
        recon_feat = torch.concat([x, output_feat], dim=1)
        out = self.refine_net(recon_feat)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder_name:str='focalnet_tiny', pretrained:bool=True):
        super().__init__()

        self.encoder = FocalNet(encoder_name, pretrained)
        self.setup_hooks()

    def setup_hooks(self):
        logging.info("\033[0;32m"+ "Setting up hooks for encoder."+ "\033[0;32m")
        self.hooks = [
            Hook(self.encoder.backbone.layers[i])
            for i in range(len(self.encoder.backbone.layers))
        ]
        logging.info("\033[0;32m"+ "Finished registering hooks"+ "\033[0;32m")

    def forward(self, x:torch.Tensor, norm:bool=True)-> None: # ingore
        return self.encoder(x, norm)


class Decoder(nn.Module):
    def __init__(self, hooks:List[Hook],
                nf:int=512,
                blur:bool=True,
                last_norm='Weight',
                num_queries=100,
                num_scales=3,
                dec_layers=9,
                ) -> None:
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)

        self.pixel_decoder = self.build_pixel_decoder()
        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)
        self.stain_decoder = MultiScaleStainDecoder(
            in_channels=[512, 512, 256],
            num_scales=num_scales,
            num_queries=num_queries,
            dec_layers=dec_layers,
        )

    def build_pixel_decoder(self):
        decoder_layers = []
        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)

    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.pixel_decoder[0](encode_feat)
        out1 = self.pixel_decoder[1](out0) 
        out2 = self.pixel_decoder[2](out1) 
        out3 = self.last_shuf(out2) 

        out = self.stain_decoder([out0, out1, out2], out3)
           
        return out


class MultiScaleStainDecoder(nn.Module):
    def __init__(self, 
                in_channels,
                hidden_dim=256,
                num_queries=100,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=9,
                pre_norm=False,
                stain_embed_dim=256,
                enforce_input_project=True,
                num_scales=3
                ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dec_layers = dec_layers
        self.pre_norm = pre_norm
        self.stain_embed_dim = stain_embed_dim
        self.enforce_input_project = enforce_input_project
        self.num_scales = num_scales
        
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_feed_forward_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0., normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0., normalize_before=pre_norm)
            )
            self.transformer_feed_forward_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0., normalize_before=pre_norm)
            )


        self.norm = nn.LayerNorm(hidden_dim)

        self.query_feat = nn.Embedding(num_embeddings=num_queries, embedding_dim=hidden_dim)
        self.query_pos = nn.Embedding(num_embeddings=num_queries, embedding_dim=hidden_dim)
        self.level_embeddings = nn.Embedding(num_embeddings=num_scales, embedding_dim=hidden_dim)

        self.input_proj = nn.ModuleList()
        for i in range(self.num_scales):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    nn.Conv2d(in_channels=in_channels[i], out_channels=hidden_dim, kernel_size=1)
                )
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)

                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        self.stainer = MLP(hidden_dim, hidden_dim, stain_embed_dim, 3)

    
    def forward(self, x:List[torch.Tensor], img_features:torch.Tensor)->torch.Tensor:
        assert len(x) == self.num_scales

        src = []
        pos = []

        for i in range(self.num_scales):
            pos.append(self.pe_layer(x[i], None).flatten(2)) # BxCxHxW->BxCxL
            # [D] -> Broadcast to [B, D, L]
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embeddings.weight[i][None, :, None])

            # BxDxL -> LxBXD
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # [N, D]->[N, 1, D]
        query_embed = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.dec_layers):
            level_index = i % self.num_scales

            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed

            )

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            output = self.transformer_feed_forward_layers[i](output)

    
        output = self.norm(output)
        output = output.transpose(0, 1) # [N, B, D] -> [B, N, D]
        stain_embed = self.stainer(output) # [B, N, D] -> [B, N, C]

        out = torch.einsum(
            'bqc, bchw->bqhw',
            stain_embed, img_features
        ) # [B, N, H, W]

        return out