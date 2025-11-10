# DDStainer: Towards Virtual Staining via Dual Decoder Network

A deep learning framework for virtual staining of histopathological images using a dual decoder architecture with FocalNet backbone.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Export Models](#export-models)

## ✨ Features

- **Dual Decoder Architecture**: Multi-scale decoder with transformer-based stain decoder
- **FocalNet Backbone**: Efficient vision transformer encoder (tiny/small/base/large variants)
- **GAN Training**: Supports adversarial training with discriminator
- **Multi-scale Loss**: Combined pixel loss, perceptual loss, and GAN loss
- **Large Image Inference**: Sliding window inference with reflection padding for arbitrary image sizes
- **HuggingFace Integration**: Export and load models via HuggingFace transformers
- **Comprehensive Metrics**: FID, PSNR, SSIM, and MAE evaluation

## 🚀 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (for GPU training)

### Install Dependencies

```bash
pip install torch torchvision torchmetrics transformers timm
pip install tqdm pillow numpy
```

## 📁 Project Structure

```
VirutalStainer/
├── Criterion/              # Loss functions
│   ├── loss_function.py   # Pixel loss, GAN loss
│   └── perceptual_loss.py # VGG-based perceptual loss
├── Dataset/                # Data loading
│   └── ImageDataset.py    # Image dataset with augmentation
├── Modeling/               # Model architecture
│   ├── Backbone/
│   │   └── FocalNet.py    # FocalNet encoder
│   ├── Discriminator/
│   │   └── discriminator.py
│   ├── modules/           # Attention, UNet, position embedding
│   └── model.py           # DDStainer main model
├── Pipeline/               # HuggingFace integration
│   └── DDStainer.py       # Pretrained model wrapper
├── trainer/                # Training framework
│   ├── train_master.py    # Base training engine
│   └── train_ddstainer.py # DDStainer trainer
├── opt.py                  # Configuration file
├── main.py                 # Training entry point
├── test.py                 # Inference script
├── benchmark.py            # Evaluation metrics
└── export_hf_models.py     # Export to HuggingFace format
```

## 🏃 Quick Start

### 1. Prepare Data

Organize your data in the following structure:

```
data/
├── lr/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── hr/
    ├── train/
    │   └── *.jpg
    └── val/
        └── *.jpg
```

### 2. Configure Training

Edit `opt.py` to set your data path and training parameters:

```python
opt['data_root'] = 'data/tiles'
opt['batch_size'] = 8
opt['iterations'] = 100000
# ... other configurations
```

### 3. Start Training

```bash
python main.py
```

## ⚙️ Configuration

The `opt.py` file contains all configuration parameters:

### Dataset Configuration
- `data_root`: Root directory of dataset
- `batch_size`: Batch size for training
- `num_workers`: Number of data loading workers

### Model Configuration
- `encoder_name`: FocalNet variant (`focalnet_tiny`, `focalnet_small`, `focalnet_base`, `focalnet_large`)
- `pretrained`: Whether to use pretrained encoder weights
- `num_queries`: Number of query tokens in transformer decoder
- `num_scales`: Number of scales in multi-scale decoder
- `dec_layers`: Number of decoder layers

### Training Configuration
- `iterations`: Total training iterations
- `init_lr`: Initial learning rate
- `valid_iterations`: Validation frequency
- `checkpoint_freq`: Checkpoint saving frequency

### Loss Configuration
- `perceptual_weight`: Weight for perceptual loss
- `gan_loss_weight`: Weight for GAN loss
- `vgg_layer_weights`: VGG layer weights for perceptual loss

## 📖 Usage

### Training

```bash
python main.py
```

Training logs and checkpoints will be saved to the path specified in `opt['save_path']`.

### Inference

For single image or folder inference:

```bash
python test.py --version <checkpoint_path> --input <input_folder> --output <output_folder>
```

The inference script supports:
- Large image processing via sliding window
- Automatic reflection padding
- Batch processing for efficiency

### Evaluation

Compute metrics between two image folders:

```bash
python benchmark.py --folder1 <reference_folder> --folder2 <generated_folder>
```

This will compute:
- **FID**: Frechet Inception Distance
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MAE**: Mean Absolute Error

### Export to HuggingFace

Export trained model to HuggingFace format:

```bash
python export_hf_models.py \
    --checkpoint_path <checkpoint.pth> \
    --config_path <opt.py> \
    --output_path <output_directory>
```

Then load the model:

```python
from Pipeline import DDStainerModel

model = DDStainerModel.from_pretrained('<output_directory>')
```

## 🔬 Model Architecture

DDStainer consists of:

1. **Encoder**: FocalNet backbone for feature extraction
2. **Pixel Decoder**: UNet-style decoder for spatial reconstruction
3. **Stain Decoder**: Multi-scale transformer decoder for stain prediction
4. **Refine Network**: Final refinement layer

The model supports:
- Multi-scale feature fusion
- Attention mechanisms
- Adversarial training with discriminator

## 📊 Evaluation Metrics

- **FID**: Measures the distribution distance between real and generated images
- **PSNR**: Pixel-level similarity metric
- **SSIM**: Structural similarity considering luminance, contrast, and structure
- **MAE**: Mean absolute error between pixel values

## 🛠️ Advanced Features

### Large Image Inference

The model supports inference on images of arbitrary size using sliding window:

```python
from trainer.train_ddstainer import train_ddstainer

trainer = train_ddstainer(opt, has_discriminator=True)
output = trainer.inference(large_image_tensor, do_normalize=True)
```

Features:
- Automatic reflection padding
- Batch processing for efficiency
- Memory-efficient sliding window

### Custom Loss Functions

The framework supports:
- Pixel loss (L1)
- Perceptual loss (VGG-based)
- GAN loss (vanilla or LSGAN)
- Multi-scale GAN loss


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgement
I acknowledge the high-quality code contributions from [Mask2Former](https://github.com/facebookresearch/Mask2Former), [DDColor](https://github.com/piddnad/DDColor/tree/master), and [APISR](https://github.com/Kiteretsu77/APISR/tree/main). I highly recommend these excellent works for universal segmentation, automatic image colorization, and anime image super-resolution.

Moreover, I appreciate the high-quality IHC virtual staining dataset [HER2MATCH](https://arxiv.org/pdf/2506.18484). If you use this dataset, please cite their work with the following BibTeX:
```
@inbook{Kl_ckner_2025,
   title={GANs vs. Diffusion Models for Virtual Staining with the HER2match Dataset},
   ISBN={9783032054722},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-032-05472-2_12},
   DOI={10.1007/978-3-032-05472-2_12},
   booktitle={Deep Generative Models},
   publisher={Springer Nature Switzerland},
   author={Klöckner, Pascal and Teixeira, José and Montezuma, Diana and Cardoso, Jaime S. and Horlings, Hugo M. and Oliveira, Sara P.},
   year={2025},
   month=sep, pages={120–130} }
```



