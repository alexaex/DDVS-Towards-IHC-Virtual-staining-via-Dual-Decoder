import torch
import torch.nn as nn
import timm

class FocalNet(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        """
            Focal Modulation Networks
            NIPS 2022

            Large: [192, 384, 768, 1536]
            Base: [128, 256, 512, 1024]
            Small: [96, 192, 384, 768]
            Tiny: [96, 192, 384, 768]
       """
        super().__init__()
    
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # normalization params
        self.register_buffer(
            name='mean',
            tensor=torch.tensor(
                [0.485, 0.456, 0.406]
            )
        )

        self.register_buffer(
            name='std',
            tensor=torch.tensor(
                [0.229, 0.224, 0.225],
            )
        )

        if model_name == 'focalnet_tiny':
            self.backbone = timm.create_model(
                model_name='focalnet_tiny_lrf.ms_in1k',
                pretrained=pretrained
            )
        elif model_name == 'focalnet_small':
            self.backbone = timm.create_model(
                model_name='focalnet_small_lrf.ms_in1k',
                pretrained=pretrained
            )
        elif model_name == 'focalnet_base':
            self.backbone = timm.create_model(
                model_name='focalnet_base_lrf.ms_in1k',
                pretrained=pretrained
            )
        elif model_name == 'focalnet_large':
            self.backbone = timm.create_model(
                model_name='focalnet_large_fl4.ms_in22k',
                pretrained=pretrained
            )
        
        else:
            raise ValueError("Invalid architecture type.")
    
    def forward(self, x:torch.Tensor, norm:bool=True):
        """
            x: input image
            norm: do normalization
        """
        if norm:
            x = (x - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        x = self.backbone.forward(x)
        return x

    
if __name__ == "__main__":
    model = FocalNet(model_name='focalnet_tiny', pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

    
