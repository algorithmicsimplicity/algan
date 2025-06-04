import torch

# DEFAULT_DEVICE will be used for creating and animating mobs, unless overrode.
DEFAULT_DEVICE = torch.device('cpu')
# DEFAULT_RENDER_DEVICE will be used for rendering mobs to images, unless overrode.
DEFAULT_RENDER_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_device(DEFAULT_DEVICE)
torch.set_default_dtype(torch.float32)
