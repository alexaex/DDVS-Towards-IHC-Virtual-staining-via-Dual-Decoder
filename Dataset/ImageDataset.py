from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision import tv_tensors
import torch

import os
from glob import glob



class ImageDataset(Dataset):
    def __init__(self, data_root:str, split='train'):
        super().__init__()

        self.split = split
        self.lr_images = sorted(glob(
            os.path.join(
                data_root,
                'lr',
                split,
                '*.jpg'
            )
        ))

        self.hr_images = sorted(glob(
            os.path.join(
                data_root,
                'hr',
                split,
                '*.jpg'
            )
        ))
  
        assert len(self.lr_images) == len(self.hr_images), "Number of LR and HR images must be the same"

        if split == 'train':
            self.augmentation = v2.Compose(
                [
                    v2.RandomHorizontalFlip(0.5),
                    v2.RandomVerticalFlip(0.5),
                    v2.ToDtype({tv_tensors.Image:torch.float}, scale=True)
                ]
            )
        else:
            self.augmentation = v2.Compose(
                [
                    v2.ToDtype({tv_tensors.Image:torch.float}, scale=True)
                ]
            )

        
    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
    
       
        lr = decode_image(self.lr_images[index])
        hr = decode_image(self.hr_images[index])
        lr, hr = tv_tensors.Image(lr), tv_tensors.Image(hr)
        lr, hr = self.augmentation(lr, hr)
        if self.split == 'train':
            top, left, h, w = self._crop_coords(lr, 256)
            lr = lr[..., top:top+h, left:left+w]
            hr = hr[..., top:top+h, left:left+w]

        ret = {
            'filename': os.path.basename(self.lr_images[index]),
            'input': lr,
            'target': hr
        }
        
        return ret

    def _crop_coords(self, img, patch_size):
        H = int(img.shape[-2])
        W = int(img.shape[-1])

        if H <= patch_size:
            top = 0
        else:
            max_top = H - patch_size
            top = int(torch.randint(0, max_top + 1, (1,)).item())

        if W <= patch_size:
            left = 0
        else:
            max_left = W - patch_size
            left = int(torch.randint(0, max_left + 1, (1,)).item())
        return top, left, patch_size, patch_size
