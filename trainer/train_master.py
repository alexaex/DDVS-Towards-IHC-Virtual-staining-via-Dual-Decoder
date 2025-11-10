import torch
import torch.nn as nn
from Criterion.loss_function import PixelLoss, GANLoss, MultiScaleGANLoss
from Criterion.perceptual_loss import PerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from Dataset.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import logging
import os
import numpy as np
import random
import math

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

class train_engine:
    def __init__(
        self,
        opt:dict,
        has_discriminator:bool=False,
    ):
        self.opt = opt
        self.has_discriminator = has_discriminator
        # DATASET
        self.data_root = opt['data_root']
        self.batch_size = opt['batch_size']
        self.num_workers = opt['num_workers']
        self.pin_memory = opt['pin_memory']
        self.shuffle = opt['shuffle']
        self.prefetch_factor = opt['prefetch_factor']
        self.persistent_workers = opt['persistent_workers']
        self.drop_last = opt['drop_last']


        # Random seed
        self.seed = opt['random_seed']

    
        # iterations
        self.total_iterations = opt['iterations']
        self.val_iterations = opt['valid_iterations']
        self.checkpoint_freq = opt['checkpoint_freq']

        self.best_fid = float('inf')
        self.best_iterations = -1

        self.current_iteration = 0
        # attributes
        self.model_name = opt['model_name']
        self.generator_prefix = opt['model_name']+'_generator_{iteration}.pth'
        self.discriminator_prefix = opt['model_name']+'_discriminator_{iteration}.pth'
        self.state_dicts = 'latest_model_state'
                

        # mixed-precision
        self.scaler = torch.amp.GradScaler() 
        
        # model
        self.net_g = self.call_model()

        if has_discriminator:
            self.net_d = self.call_discriminator()

        # optimizer
        self.learning_rate = opt['init_lr']
        self.optimizer_g = torch.optim.Adam(
            params=self.net_g.parameters(),
            lr=self.learning_rate,
            betas=(opt['adam_beta1'], opt['adam_beta2'])
        )

        if has_discriminator:
            self.optimizer_d = torch.optim.Adam(
                params=self.net_d.parameters(),
                lr=self.learning_rate,
                betas=(opt['adam_beta1'], opt['adam_beta2'])
            )

        self.writer = SummaryWriter()
        self.log_steps = opt['log_steps']
        self.set_seed(self.seed)
        self.init_loss()

    def call_model(self)->nn.Module:
        pass
  
    def call_discriminator(self)->nn.Module:
        pass

    def init_loss(self):
        pass

    def calculate_loss(self, gen_hr:torch.Tensor, hr:torch.Tensor)->torch.Tensor:
        pass
    
    def run(self):
        logging.info(f"Training {self.model_name} started.")
        # print(self.opt)

        train_loader = DataLoader(ImageDataset(self.data_root, split='train'), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=self.drop_last)
        val_loader = DataLoader(ImageDataset(self.data_root, split='val'), batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=self.drop_last)
        dataset_length = len(os.listdir(os.path.join(self.data_root, 'lr', 'train')))
        start_epoch = self.current_iteration // math.ceil(dataset_length / self.batch_size)
        n_epochs = self.total_iterations // math.ceil(dataset_length / self.batch_size)
        
        iteration_index = self.current_iteration
        self.adjust_learning_rate(iteration_index)

        for epoch in range(start_epoch, n_epochs):
            logging.info(f"Epoch {epoch}: Start iteration: {iteration_index}, Learning Rate: {self.optimizer_g.param_groups[0]['lr']}")

            total_loss_per_batch = 0.
            self.net_g.train()
            for batch_idx, data in enumerate(train_loader):
                img_lr = data['input'].cuda()
                img_hr = data['target'].cuda()
                
                self.net_g_pixel_loss = 0.
                self.net_g_perceptual_loss = 0.
                self.net_g_gan_loss = 0.
                self.net_g_total_loss = 0.
                self.real_score = 0.
                self.fake_score = 0.
                self.single_iteration(img_lr, img_hr)

                total_loss_per_batch += self.net_g_total_loss.item()
                
                iteration_index += 1

                if iteration_index % self.opt['decay_iteration'] == 0:
                    self.adjust_learning_rate(iteration_index)    # adjust the learning rate to the desired one
                    print("Update the learning rate to {} at iteration {} ".format(self.optimizer_g.param_groups[0]['lr'], iteration_index))

                if iteration_index % self.log_steps == 0:
                    iteration_loss = {
                        'pix_l': self.net_g_pixel_loss.item(),
                        'percep_l': self.net_g_perceptual_loss.item(),
                        'gan_l': self.net_g_gan_loss.item(),
                        'total_l': self.net_g_total_loss.item(),
                        'real_score': self.real_score.item(),
                        'fake_score': self.fake_score.item(),
                    }
                    self.tensorboard_iteration_draw(iteration_loss, iteration_index)
                    logging.info(f"Iteration {iteration_index}: Pixel Loss: {iteration_loss['pix_l']:.3f}, Perceptual Loss: {iteration_loss['percep_l']:.3f}, GAN Loss: {iteration_loss['gan_l']:.3f}, Total Loss: {iteration_loss['total_l']:.3f}")
            
                if iteration_index % self.val_iterations == 0:
                    logging.info(f"Iteration {iteration_index}: Evaluating the model.")
                    metrics = self.evaluate(val_loader)
                    self.evaluation_log(metrics, iteration_index)
                    if metrics['fid'] < self.best_fid:
                        self.best_fid = metrics['fid'].item()
                        self.best_iterations = iteration_index
                        self.save_weight(iteration_index)
                        logging.info(f"New best FID: {self.best_fid:.3f} at iteration {iteration_index}")
                
                if iteration_index % self.checkpoint_freq == 0:
                    self.save_status(iteration_index)
                torch.cuda.empty_cache()
                
            self.save_status(iteration_index)
            self.tensorboard_epoch_draw(total_loss_per_batch / len(train_loader), epoch)

            
        self.writer.close()
         

    def adjust_learning_rate(self, iteration_idx):
        self.learning_rate = self.opt['init_lr']
        end_iteration = self.opt['iterations']

        # Calculate a learning rate we need in real-time based on the iteration_idx
        for idx in range(min(end_iteration, iteration_idx)//self.opt['decay_iteration']):
            idx = idx+1
            if idx * self.opt['decay_iteration'] in self.opt['double_milestones']:
                # double the learning rate in milestones
                self.learning_rate = self.learning_rate * 2
            else:
                # else, try to multiply decay_gamma (when we decay, we won't upscale)
                self.learning_rate = self.learning_rate * self.opt['decay_gamma']     # should be divisible in all cases

        # Change the learning rate to our target
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.learning_rate
        
        if self.has_discriminator:
            # print("We didn't yet handle discriminator, but we think that it should be necessary")
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = self.learning_rate

        assert(self.learning_rate == self.optimizer_g.param_groups[0]['lr'])

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
                
                gen_hr = self.net_g(img_lr)
                if idx == 0:
                    self._visualize_batch(img_lr, img_hr, gen_hr)
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


    def single_iteration(self, img_lr:torch.Tensor, img_hr:torch.Tensor):
        ##################### Generator #####################
        self.optimizer_g.zero_grad()
        if self.has_discriminator:
            for p in self.net_d.parameters():
                p.requires_grad = False
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen_hr = self.net_g(img_lr)
            self.calculate_loss(gen_hr, img_hr)
        self.scaler.scale(self.net_g_total_loss).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
   
    
        
        ##################### Discriminator #####################
        if self.has_discriminator:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
               real_d_preds = self.net_d(img_hr.detach().clone())
               fake_d_preds = self.net_d(gen_hr.detach().clone())

               l_d_real = self.cri_gan(real_d_preds, True, True)
               l_d_fake = self.cri_gan(fake_d_preds, False, True)
               l_d_total = l_d_real + l_d_fake

            self.scaler.scale(l_d_total).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()

            # Log real/fake score
            self.real_score = l_d_real.mean()
            self.fake_score = l_d_fake.mean()
    

    def set_seed(self, seed:int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    

    def pixel_loss(self):
        logging.info('Pixel Loss is enabled.')
        self.cri_pix = PixelLoss(weight=self.opt['L1_weight']).cuda()
    
    def perceptual_loss(self):
        logging.info("Perceptual Loss is enabled")
        self.cri_perceptual = PerceptualLoss(layer_weights=self.opt['vgg_layer_weights'],
                                             vgg_type=self.opt['vgg_type'],
                                             perceptual_weight=self.opt['perceptual_weight'],
                                            ).cuda() 
    
    def adversarial_loss(self):
        logging.info("GAN Loss is enabled")
        if self.opt['discriminator_type'] == 'PatchDiscriminator':
            self.cri_gan = MultiScaleGANLoss(gan_type='lsgan', loss_weight=self.opt['gan_loss_weight']).cuda()
            logging.info(f"MultiScaleGANLoss is enabled with loss weight {self.opt['gan_loss_weight']}")
        else:
            self.cri_gan = GANLoss(gan_type='vanilla', loss_weight=self.opt['gan_loss_weight']).cuda()
            logging.info(f"GANLoss is enabled with loss weight {self.opt['gan_loss_weight']}")
    
    def tensorboard_epoch_draw(self, epoch_loss, epoch):
        self.writer.add_scalar('Loss/Epoch_total', epoch_loss, epoch)

    def tensorboard_iteration_draw(self, iteration_loss, iteration):
        self.writer.add_scalar('Loss/Pixel', iteration_loss['pix_l'], iteration)
        self.writer.add_scalar('Loss/Perceptual', iteration_loss['percep_l'], iteration)
        self.writer.add_scalar('Loss/GAN', iteration_loss['gan_l'], iteration)
        self.writer.add_scalar('Loss/Total', iteration_loss['total_l'], iteration)
        self.writer.add_scalars('Discriminator/Score', {
            'Real': iteration_loss['real_score'],
            'Fake': iteration_loss['fake_score'],
        }, iteration)

    
    def evaluation_log(self, metrics, iteration):
        self.writer.add_scalar('Metrics/FID', metrics['fid'], iteration)
        self.writer.add_scalar('Metrics/PSNR', metrics['psnr'], iteration)
        self.writer.add_scalar('Metrics/SSIM', metrics['ssim'], iteration)
        self.writer.add_scalar('Metrics/MAE', metrics['mae'], iteration)

    def best_performance_log(self, metrics, iteration):
        self.writer.add_scalar('Best/FID', metrics['fid'], iteration)
        self.writer.add_scalar('Best/PSNR', metrics['psnr'], iteration)
        self.writer.add_scalar('Best/SSIM', metrics['ssim'], iteration)
        self.writer.add_scalar('Best/MAE', metrics['mae'], iteration)

    def save_weight(self, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.net_g.state_dict(),
             },
        os.path.join(self.opt['save_path'], self.generator_prefix.format(iteration=iteration))
        )
        logging.info(f"Best Model saved to {os.path.join(self.opt['save_path'], self.generator_prefix.format(iteration=iteration))}")

    def save_status(self, iteration):
        status = {}
        status['iteration'] = iteration
        status['model_state_dict'] = self.net_g.state_dict()
        status['optimizer_g_state_dict'] = self.optimizer_g.state_dict()
        status['scaler_state_dict'] = self.scaler.state_dict()
        if self.has_discriminator:
            status['discriminator_state_dict'] = self.net_d.state_dict()
            status['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
        torch.save(status, os.path.join(self.opt['save_path'], self.state_dicts))
        logging.info(f"Latest Model and optimizer states saved to {os.path.join(self.opt['save_path'], self.state_dicts)}")


    def resume(self, status:str):
        checkpoint = torch.load(status)
        self.current_iteration = checkpoint['iteration']
        self.net_g.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.has_discriminator:
            self.net_d.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        logging.info(f"Resume training from {status}")
    
    def _visualize_batch(self, img_lr, img_hr, gen_hr, iteration=None):
        """
        Visualize images from a batch: LR input, generated HR, and ground truth HR.
        Randomly select a subset of images from the batch for visualization.
        
        Args:
            img_lr: Low-resolution input images [B, C, H, W]
            img_hr: High-resolution ground truth images [B, C, H, W]
            gen_hr: Generated high-resolution images [B, C, H, W]
            iteration: Current iteration number for file naming
        """
        batch_size = img_lr.shape[0]
        
        # Randomly select indices to visualize (max 4 images)
        num_vis = min(4, batch_size)
        selected_indices = random.sample(range(batch_size), num_vis)
        
        # Create visualization directory
        vis_dir = os.path.join(self.opt['save_path'], 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Select images to visualize
        img_lr_vis = img_lr[selected_indices].cpu()
        img_hr_vis = img_hr[selected_indices].cpu()
        gen_hr_vis = gen_hr[selected_indices].cpu()
        
        # Create grid: for each selected image, show LR, Generated, GT in a row
        vis_images = []
        for i in range(num_vis):
            vis_images.append(img_lr_vis[i])      # LR
            vis_images.append(gen_hr_vis[i])      # Generated
            vis_images.append(img_hr_vis[i])      # Ground Truth
        
        # Create grid: 3 columns (LR, Generated, GT) x num_vis rows
        grid = make_grid(vis_images, nrow=3, normalize=True, pad_value=1)
        
        # Save visualization
        if iteration is not None:
            save_path = os.path.join(vis_dir, f'visualization_iter_{iteration}.png')
        else:
            save_path = os.path.join(vis_dir, 'visualization.png')
        
        save_image(grid, save_path)
        logging.info(f"Visualization saved to {save_path}")