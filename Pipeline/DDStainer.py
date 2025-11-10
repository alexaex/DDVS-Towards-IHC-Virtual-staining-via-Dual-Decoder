from Modeling import DDStainer
from transformers import PretrainedConfig, PreTrainedModel
from typing import Tuple, Dict, List
import torch


class DDStainerConfig(PretrainedConfig):
    model_type = 'DDStainer'

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
            **kwargs
            ) -> None:

        super().__init__()

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.num_input_channels = num_input_channels
        self.input_size = input_size
        self.nf = nf
        self.num_output_channels = num_output_channels
        self.last_norm = last_norm
        self.num_queries = num_queries
        self.num_scales = num_scales
        self.dec_layers = dec_layers


class DDStainerModel(PreTrainedModel):
    config_class = DDStainerConfig
    
    def __init__(self, config: DDStainerConfig):
        super().__init__(config)
        self.config = config
        # Initialize model on CPU first to avoid meta device issues
        # The model will be moved to the appropriate device later via .to() or .cuda()
        with torch.device('cpu'):
            self.model = DDStainer(
                encoder_name=config.encoder_name,
                pretrained=config.pretrained,
                num_input_channels=config.num_input_channels,
                input_size=config.input_size,
                nf=config.nf,
                num_output_channels=config.num_output_channels,
                last_norm=config.last_norm,
                num_queries=config.num_queries,
                num_scales=config.num_scales,
                dec_layers=config.dec_layers
            )
        self.post_init()

    def forward(self, x: torch.Tensor, do_normalize: bool = True) -> torch.Tensor:
        return self.model(x, do_normalize)

    