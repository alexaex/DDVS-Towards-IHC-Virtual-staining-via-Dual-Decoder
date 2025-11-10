from .train_master import train_engine
from Modeling import DDStainer
from Modeling import MultiScaleDiscriminator
import torch.nn as nn
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


class train_ddstainer(train_engine):
    def __init__(self, opt:dict, has_discriminator:bool=False):
        super().__init__(opt, has_discriminator)

    def call_model(self)->nn.Module:
        return DDStainer(
            encoder_name=self.opt['encoder_name'],
            pretrained=self.opt['pretrained'],
            num_input_channels=self.opt['num_input_channels'],
            input_size=self.opt['input_size'],
            nf=self.opt['nf'],
            num_output_channels=self.opt['num_output_channels'],
            last_norm=self.opt['last_norm'],
            num_queries=self.opt['num_queries'],
            num_scales=self.opt['num_scales'],
            dec_layers=self.opt['dec_layers'],
        ).cuda()

    def call_discriminator(self)->nn.Module:
        return MultiScaleDiscriminator(3).cuda()

    def init_loss(self):
        self.pixel_loss()
        self.perceptual_loss()
        self.adversarial_loss()

    def calculate_loss(self, gen_hr:torch.Tensor, hr:torch.Tensor):
        self.net_g_pixel_loss = self.cri_pix(gen_hr, hr)

        self.net_g_perceptual_loss = self.cri_perceptual(gen_hr, hr)

        fake_g_preds = self.net_d(gen_hr)
        self.net_g_gan_loss = self.cri_gan(fake_g_preds, True, False)
        self.net_g_total_loss = self.net_g_pixel_loss + self.net_g_perceptual_loss + self.net_g_gan_loss
    
    # overload evluation func
    @torch.no_grad()
    def evaluate(self, loader):
        self.net_g.eval()
        
        fid_helper = FrechetInceptionDistance(feature=2048, normalize=True).to('cuda')
        psnr_helper = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
        ssim_helper = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        mae_helper = MeanAbsoluteError().to('cuda')

        for idx, data in tqdm(enumerate(loader), total=len(loader)):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                img_lr = data['input'].to('cuda')
                img_hr = data['target'].to('cuda')
                
                gen_hr = self.inference(img_lr, do_normalize=True)
                if idx % 1000 == 0:
                    self._visualize_batch(img_lr, img_hr, gen_hr, iteration=idx)
                psnr_helper.update(gen_hr, img_hr)
                ssim_helper.update(gen_hr, img_hr)
                mae_helper.update(gen_hr, img_hr)

                img_hr = nn.functional.interpolate(
                    img_hr,
                    size=(299, 299),
                    mode='bilinear',
                    align_corners=False,
                )
                gen_hr = nn.functional.interpolate(
                    gen_hr,
                    size=(299, 299),
                    mode='bilinear',
                    align_corners=False,
                )
               

                fid_helper.update(img_hr, real=True)
                fid_helper.update(gen_hr, real=False)
          
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            metrics = {
                'fid': fid_helper.compute(),
                'psnr': psnr_helper.compute(),
                'ssim': ssim_helper.compute(),
                'mae': mae_helper.compute(),
            }

        del fid_helper, psnr_helper, ssim_helper, mae_helper
        torch.cuda.empty_cache()
        return metrics
    
    def print_model(self):
        from torchinfo import summary
        summary(self.net_g, [1, 3, 256, 256])
        summary(self.net_d, [1, 3, 256, 256])
    
    def perf_probing(self):
        from fvcore.nn import FlopCountAnalysis
        flop_count = FlopCountAnalysis(self.net_g, (torch.randn([1, 3, 256, 256])).cuda())
        print(f"GFLOPS: {flop_count.total() / 1e9}")

   
    def inference(self, x:torch.Tensor, do_normalize:bool=True)->torch.Tensor:

        """
            a trick is used for sliding window.
            with fold and unfold, convlution operator can be performed by GEMM.
            actually, fold and unfold is a kind of im2col and col2im.
        """

        assert x.size(0) == 1, "Batch size must be 1 for inference"
        

        window_size = 256
        stride = 256
        
        _, C, H, W = x.shape
    
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        _, _, H_padded, W_padded = x.shape
        
        x_unfolded = nn.functional.unfold(x, kernel_size=window_size, stride=stride)
        
        num_patches = x_unfolded.shape[2]
        x_patches = x_unfolded.view(1, C, window_size, window_size, num_patches)
        x_patches = x_patches.permute(4, 0, 1, 2, 3).contiguous()
        x_patches = x_patches.view(num_patches, C, window_size, window_size)
        
        # Batch inference: process all patches at once
        self.net_g.eval()
        with torch.no_grad():
            x_patches = x_patches.cuda()

            output_patches = self.net_g(x_patches, do_normalize=True)
        
        C_out = output_patches.shape[1]
        
        # Reshape for fold operation: [num_patches, C_out, window_size, window_size] -> [1, C_out*window_size*window_size, num_patches]
        output_patches = output_patches.permute(1, 2, 3, 0).contiguous()  
        output_patches = output_patches.view(C_out * window_size * window_size, num_patches) 
        output_patches = output_patches.unsqueeze(0)

        output = nn.functional.fold(
            output_patches,
            output_size=(H_padded, W_padded),
            kernel_size=window_size,
            stride=stride
        )
        
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, pad_top:H_padded-pad_bottom, pad_left:W_padded-pad_right]
        
        return output
