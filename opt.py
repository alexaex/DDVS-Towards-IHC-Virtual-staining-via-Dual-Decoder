# opt.py
# Configuration dictionary for training DDStainer model

opt = {}

# ==================== Dataset Configuration ====================
opt['data_root'] = 'data/tiles'
opt['batch_size'] = 8
opt['num_workers'] = 4
opt['pin_memory'] = True
opt['shuffle'] = True
opt['prefetch_factor'] = 2
opt['persistent_workers'] = True
opt['drop_last'] = True

# ==================== Training Configuration ====================
opt['random_seed'] = 2025
opt['iterations'] = 100000
opt['valid_iterations'] = 4000
opt['checkpoint_freq'] = 10000
opt['log_steps'] = 100

# ==================== Model Configuration ====================
opt['model_name'] = 'DDStainer'
opt['save_path'] = './logs/DDStainer/v1'

# ==================== Optimizer Configuration ====================
opt['init_lr'] = 2e-4
opt['adam_beta1'] = 0.9
opt['adam_beta2'] = 0.999

# ==================== Learning Rate Schedule ====================
opt['decay_iteration'] = 20000
opt['decay_gamma'] = 0.5
opt['double_milestones'] = []

# ==================== DDStainer Model Configuration ====================
opt['encoder_name'] = 'focalnet_base'
opt['pretrained'] = True
opt['num_input_channels'] = 3
opt['input_size'] = (256, 256)
opt['nf'] = 512
opt['num_output_channels'] = 3
opt['last_norm'] = 'Weight'
opt['num_queries'] = 256
opt['num_scales'] = 3
opt['dec_layers'] = 9

# ==================== Loss Function Configuration ====================
opt['vgg_type'] = 'vgg19'
opt['vgg_layer_weights'] = {'conv1_2': 0.0625, 'conv2_2': 0.125, 'conv3_4': 0.25, 'conv4_4': 0.5, 'conv5_4': 1.0}

opt['L1_weight'] = 5.
opt['perceptual_weight'] = 0.05
opt['discriminator_type'] = 'PatchDiscriminator'
opt['gan_loss_weight'] = 1.
