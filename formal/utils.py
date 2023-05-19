import colorsys
import numpy as np
import torch


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    j = 0
    step = 360.0 / num
    while j < num:
        h = i
        s = 90 #+ random.random() * 10
        l = 50 #+ random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
        j += 1
    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors, hls_colors


def cpunumpy(x):
    return [_.detach().cpu().numpy() for _ in x]
    
class recorder_vector(object): # save the object value
    def __init__(self, keys=None):
        self.last=0
        self.values=[]
        self.nums=[]
        self.keys = keys
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        self.values = np.asarray(self.values)
        self.nums = np.asarray(self.nums)
        if len(self.values.shape) > 2:
            self.values = self.values.squeeze()
        sum=(np.expand_dims(self.nums, 1) * self.values).sum(0)
        count=np.sum(self.nums)
        if self.keys:
            return dict(zip(self.keys, list(sum/count)))
        else:
            return sum/count
        

def checkpoint(j, model, args):
    torch.save(model, './tensorboard/%s/%s_%d.checkpoint' % (args.comment, args.style, j))