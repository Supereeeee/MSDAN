from thop import profile
import torch
from .EFEN_arch import EFEN

net = EFEN(channels=48, num_DFEB=8, upscale_factor=4)
input = torch.randn(1, 3, 320, 180)
flops, params = profile(net, (input,))
print('flops[G]: ', flops/1e9, 'params[K]: ', params/1e3)

#-----------------------------------(EBFB+EMSA)*8-----------------------------------------#
# FLOPs:  20597114880.0 = 20.6G     # params:  227358.0 = 227.4K  (x4)(320*180)(48 channel)
# FLOPs:  33954855840.0 = 34.0G     # params:  227418.0 = 227.4K  (x3)(426*240)(48 channel)
# FLOPs:  75678451200.0 = 75.7G     # params:  227358.0 = 227.4K  (x2)(640*360)(48 channel)
