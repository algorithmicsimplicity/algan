import torch
import torch.nn.functional as F


#TODO fix up this code
def bloom_filter_old(x, blur_width=0.01*0.0005, num_iterations=3, kernel_size=31, strength=10, scale_factor=8):
#def bloom_filter(x, blur_width=0.01*0.0005, num_iterations=3, kernel_size=11, strength=10, scale_factor=8):
    #kernel_size = int(kernel_size * x.shape[-3] / 2160)
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype
    x = x.to(torch.float) / 255
    x[...,-1] = (x[...,-1]) * strength
    xb = torch.cat((x[...,:-1].clamp(min=1/255) * x[...,-1:], x[...,-1:]), -1)

    #xb = torch.cat((x[...,:-1].clamp(min=1/255) * (1-x[...,-1:]).clamp_(min=0, max=1) + x[...,-1:].clamp(min=0, max=1) * torch.ones_like(x[...,:-1].clamp(min=1/255)), x[...,-1:]), -1)
    #xb = torch.cat((x[...,:-1].clamp(min=1/255), x[...,-1:]), -1)
    #d = kernel_size / (min(x.shape[0], x.shape[1])/scale_factor)
    #filter = torch.exp(-1*(torch.linspace(-d, d, kernel_size, device=x.device)**2) * 2 / blur_width)
    d = 1
    filter = torch.exp(-1*(torch.linspace(-d, d, kernel_size, device=x.device)**2))
    filter /= filter.sum()
    filter *= 1
    #filter /= filter.amax()
    filter_horizontal = filter.view(1, 1,1,kernel_size).expand(xb.shape[-1],-1,-1,-1)
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)
    #counter_horizontal = torch.ones_like(filter_horizontal)
    #counter_horizontal = counter_horizontal / counter_horizontal.numel()
    #counter_vertical = torch.ones_like(filter_vertical)
    #counter_vertical = counter_vertical / counter_vertical.numel()
    #count = x[...,-1:].expand(-1,-1,3)/255
    p = (kernel_size-1)//2
    xb = xb.permute(-1,0,1)
    orig_shape = xb.shape[-2:]
    xb = F.interpolate(xb.unsqueeze(0), scale_factor=1/scale_factor, mode='bilinear').squeeze(0)
    dists = torch.stack((torch.linspace(-1, 1, kernel_size, device=x.device).view(-1,1).expand(-1,kernel_size),
                       torch.linspace(-1, 1, kernel_size, device=x.device).view(1,-1).expand(kernel_size, -1)), -1)
    dists = dists.square().sum(-1, keepdim=True).unsqueeze(-1)

    k = 1#kernel_size * kernel_size * 0.01
    #count = count.permute(-1,0,1)
    for i in range(num_iterations):
        """xbu = F.unfold(xb.unsqueeze(0), (kernel_size, kernel_size), padding=(p, p)).squeeze(0)
        xbu = unsquish(unsquish(unsquish(xbu, 0, -xb.shape[0]), -1, xb.shape[-1]), 1, kernel_size)
        #a = torch.exp(-dists) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists*2) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / (xbu[-1:]).clamp(min=1e-5)) * (1 - (dists)).clamp(min=0,max=1) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / (xbu[-1:]).clamp(min=1e-5)) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / 2) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        a = torch.exp(-dists / 2) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        t = a.clamp(min=0, max=1)
        a[:,p,p] += k#kernel_size*kernel_size*0.3
        n = a.sum((1,2), keepdim=True).clamp(min=1e-5)
        a = a / n
        a[:,p,p] = 0
        q = (1.2*a.sum((1,2))).clamp(min=0, max=1)
        #xb = torch.cat((((xbu[:-1] * (1-t) + t * torch.ones_like(xbu[:-1])) * a).sum((1,2)), (n).sum((1,2))), 0)
        xb = torch.cat((((xbu[:-1]) * a).sum((1,2)), (n).sum((1,2))), 0)
        xb[:-1] = xb[:-1]*(1-q) + (q * torch.ones_like(xb[:-1]))
        continue"""
        xb = F.conv2d(xb, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        xb = F.conv2d(xb, filter_vertical, padding=(p, 0), groups=xb.shape[0])
        #count = F.conv2d(count, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        #count = F.conv2d(count, filter_vertical, padding=(p, 0), groups=xb.shape[0])
        #n2 = xb[-1:] + k
        #if (xb[-1:].amax() <= 1):
        #    break

    xb = F.interpolate(xb.unsqueeze(0), size=orig_shape, mode='bilinear').squeeze(0)
    xb = xb.permute(1,2,0)


    a = xb[...,-1:].clamp(min=0, max=1)
    a2 = (xb[..., -1:] * 0.5).clamp(min=0, max=1)
    m = ((xb[...,-1:] - x[...,-1:]) >= 0).float()
    a4 = torch.zeros_like((xb[...,-1:] - x[...,-1:]).clamp(min=0, max=1))
    r = 0.5
    #a5 = ((1/r)*((xb[...,-1:] +1).log() - r)).clamp(min=0, max=1)
    a5 = ((xb[...,-1:] +1).log() / 3).clamp(min=0, max=1)
    #a5 = (((xb[...,-1:] / x[...,-1:].clamp(min=1e-5)))).clamp(min=0, max=1)
    #a5 = (1-((xb[...,-1:] - x[...,-1:]))).clamp(min=0, max=1)
    #a3 = (((xb[...,-1:] - x[...,-1:]) * 0.5) + 0.5).clamp(min=0,max=1)

    """xb[...,:-1] = xb[...,:-1] * (1-a2) + a2 * torch.ones_like(xb[...,:-1])
    xb = xb.permute(-1, 0, 1)
    xb = F.interpolate(xb.unsqueeze(0), scale_factor=1 / scale_factor, mode='bilinear').squeeze(0)
    for i in range(num_iterations):
        xb = F.conv2d(xb, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        xb = F.conv2d(xb, filter_vertical, padding=(p, 0), groups=xb.shape[0])
    xb = F.interpolate(xb.unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze(0)
    xb = xb.permute(1,2,0)"""

    xb = ((xb[..., :-1] + x[..., :-1] * (x[..., -1:] + k)) / (xb[..., -1:] + x[..., -1:] + k)) * (1-a5) + a5 * torch.ones_like(xb[...,:-1])
    #xb = (xb[...,:-1] + x[...,:-1] * (x[...,-1:]+k)) / (xb[...,-1:]+x[...,-1:]+k)# * m + (1-m) * (xb[...,-1:])# + a4 * torch.ones_like(xb[...,:-1]))
    #xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1] * (x[...,-1:]+k))# * m + (1-m) * (xb[...,-1:])# + a4 * torch.ones_like(xb[...,:-1]))
    #xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1]) * (1-a3) + a3 * torch.ones_like(xb[...,:-1])
    #xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1])

    #count = count.permute(1,2,0) + 1

    n3 = xb[..., -1:] + k
    #xb = (xb[...,:-1] + x[...,:-1] * k) / n3# / xb[...,-1:].clamp(min=1e-5)#*strength
    #xb = xb[...,:-1] + (x[...,:-1] * (k + x[...,-1:])) / xb[...,-1:].clamp(min=1e-5)
    #glow = (glow / glow.amax().clamp_(min=255)) * 255
    #glow = glow.clamp(max=255)
    a = (xb[...,-1:]).clamp(min=0, max=1)
    #glow = glow * (1-a) + a * (torch.ones_like(glow))# * 0.4 + glow * 0.6)
    #out = (x[...,:-1] * (1-a) + a * glow)# / (1+xb[...,-1:])
    a2 = (xb[...,-1:]*1).clamp(min=0, max=1)
    out = xb#+ (x[..., :-1])# + glow)  # / (1+xb[...,-1:])
    #out = (x[..., :-1] + glow)  # / (1+xb[...,-1:])
    #out = (x[...,:-1] * (1-a) + (a2) * glow)# / (1+xb[...,-1:])
    #out = (x[...,:-1] *(1-a) + a * glow)# / (1+xb[...,-1:])
    return (((out * 255).clamp_(max=255))).to(xdtype)
    #return ((out / out.amax().clamp_(min=255)) * 255).to(xdtype)
    #return (x[...,:-1] + xb*strength).clamp_(max=255).to(xdtype)


def bloom_filter(x, num_iterations=3, kernel_size=31, strength=10, scale_factor=8):
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype

    x = x.to(torch.float) / 255
    color = x[...,:-1]
    glow = x[..., -1:] * strength

    color = color * glow
    # To allow for dark colors to bloom as well as bright, we apply bloom filter to both color and inverse color,
    # then average the result at the end.
    #color = torch.cat((color*glow, (1-color)*(glow), glow), -1)

    d = 1
    filter = torch.exp(-1*(torch.linspace(-d, d, kernel_size, device=x.device)**2))
    filter /= filter.sum()
    filter_horizontal = filter.view(1, 1,1,kernel_size).expand(color.shape[-1],-1,-1,-1)
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)

    # Do gaussian convolutions to spread colors.
    p = (kernel_size-1)//2

    color = color.permute(-1,0,1)
    orig_shape = color.shape[-2:]
    color = F.interpolate(color.unsqueeze(0), scale_factor=1/scale_factor, mode='bilinear').squeeze(0)

    for i in range(num_iterations):
        color = F.conv2d(color, filter_horizontal, padding=(0, p), groups=color.shape[0])
        color = F.conv2d(color, filter_vertical, padding=(p, 0), groups=color.shape[0])

    color = F.interpolate(color.unsqueeze(0), size=orig_shape, mode='bilinear').squeeze(0)
    color = color.permute(1,2,0)

    out = color + x[...,:-1]
    return (out * 255).clamp_(min=0, max=255).to(xdtype)

    color = color[...,:-1]

    s = color.shape[-1]//2

    color = color + torch.cat((x[...,:-1], 1-x[...,:-1]), -1)

    #color = torch.maximum(color[...,:s], 1-color[...,s:])
    #inverse_color = color[...,s:]
    # Take average of both color and inverse color.
    color = color[...,:s]
    #m = color[...,:s].norm(p=2,dim=-1,keepdim=True) > color[...,s:].norm(p=2,dim=-1,keepdim=True)
    #color = (color[...,:s] * m + (~m) * (1-color[...,s:]))
    #color = (1-color[...,s:])
    #out = x[...,:-1] + color
    out = color
    # Interpolate original color and bloomed color based on
    # how much glow was accumulated.
    #w = 1/(1+glow)
    #out = x[...,:-1] * w + (1-w) * color

    return (out * 255).clamp_(min=0, max=255).to(xdtype)