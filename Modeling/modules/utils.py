import torch.nn as nn


class Hook:
    feature = None

    def __init__(self, module:nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output
    
    def remove_hook(self):
        self.hook.remove()