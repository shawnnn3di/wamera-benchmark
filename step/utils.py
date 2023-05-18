import argparse
import numpy as np
import torch


class Recorder(object): # save the object value
    def __init__(self):
        self.last=0
        self.values=[]
        self.nums=[]
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        sum=np.sum(np.asarray(self.values) * np.expand_dims(np.asarray(self.nums), (1, 2)), 0)
        count=np.sum(np.asarray(self.nums))
        return sum/count

def l1loss(x, y):
    # loss = torch.tensor(0.0).to(x[0].device)
    # for i in range(len(x)):
    #     metric = torch.mean(torch.abs(x[i] - y[i]))
    #     loss += metric
    loss = torch.abs(x.squeeze() - y)
    return torch.sum(loss)#, metric

def l2loss(x, y, k=1, b=1):
    # loss = torch.tensor(0.0).to(x[0].device)
    # for i in range(len(x)):
    #     metric = torch.mean(torch.abs(x[i] - y[i]) ** 2)
    #     loss += metric
    loss = torch.sum((x - y) ** 2) / x.shape[0]
    return loss#, metric


def matthewloss(x, y, k=1, b=1):
    # x = x.permute([0, 2, 1, 3, 4])
    # y = torch.tensor(y).to(x.device)
    # loss = torch.tensor(0.0).to(x[0].device)
    # for i in range(len(x)):
    #     metric = torch.sum((torch.abs(x[i] - y[i]) ** 2) * (y[i] * k + ((y[i] > 0) * 2 - 1) * b))
    #     loss += metric
    # loss = (x - y) ** 2 * (y * k + b * ((y > 0) * 2 - 1))
    loss = torch.sum(((x.squeeze() - y)) ** 2 * (y.abs() * k + b))# + (x.abs() > 0.05).sum() * 0.00001
    return loss

def forward_direct(teacher, translator, student, batch, halving, device):
    fake = translator(halving(batch['csi']))
    real = torch.cat([_.unsqueeze(0) for _ in halving(batch['img'].to(device))], 0)

    out_kpt_fake, out_aff_fake, out_kpt_real, out_aff_real = [], [], [], []
    # for i in range(real.shape[0]):
    #     out_kpt_r, out_aff_r, ntmd_real = teacher(real[i])
    #     out_kpt_f, out_aff_f, ntmd_fake = teacher(fake[i].permute(1, 0, 2, 3))

    #     out_kpt_fake.append(out_kpt_f)
    #     out_aff_fake.append(out_aff_f)
    #     out_kpt_real.append(out_kpt_r)
    #     out_aff_real.append(out_aff_r)
        
    fake_ = torch.cat([_ for _ in fake.permute(0, 2, 1, 3, 4)])
    real_ = torch.cat([_ for _ in real])
    out_kpt_r, out_aff_r, ntmd_real = teacher(real_)
    out_kpt_f, out_aff_f, ntmd_fake = teacher(fake_)

    # out_kpt_fake = torch.stack(out_kpt_fake)
    # out_aff_fake = torch.stack(out_aff_fake)
    # out_kpt_real = torch.stack(out_kpt_real)
    # out_aff_real = torch.stack(out_aff_real)

    # return out_kpt_real, out_aff_real, out_kpt_fake, out_aff_fake
    return out_kpt_r, out_aff_r, out_kpt_f, out_aff_f


def forward_skip(teacher, translator, student, batch, halving, device):
    out_kpt_fake, out_aff_fake, _ = student(halving(batch['csi']))
    real = torch.cat([_.unsqueeze(0) for _ in halving(batch['img'].to(device))], 0)

    out_kpt_fake, out_aff_fake, out_kpt_real, out_aff_real = [], [], [], []
    for i in range(real.shape):
        out_kpt_r, out_aff_r, ntmd_real = teacher(real[i])

        out_kpt_real.append(out_kpt_r)
        out_aff_real.append(out_aff_r)

    out_kpt_real = torch.stack(out_kpt_real)
    out_aff_real = torch.stack(out_aff_real)

    return out_kpt_real, out_aff_real, out_kpt_fake, out_aff_fake


def add_args(parser: argparse.ArgumentParser):
    # parser.add_argument('--mode',           default='direct')
    # parser.add_argument('--correspond',     default='many2many')
    parser.add_argument('--batch_size',     default=32, type=int)
    parser.add_argument('--shuffle_train',  default=True, type=bool)
    parser.add_argument('--num_workers',    default=0, type=int)
    parser.add_argument('--lr',             default=1e-2, type=float)
    parser.add_argument('--weight_decay',   default=1e-5, type=float)
    parser.add_argument('--num_epoch',      default=300, type=int)
    parser.add_argument('--half',           default=False, type=bool)
    parser.add_argument('--valid_gap',      default=10, type=int)
    parser.add_argument('--preview_gap',    default=100, type=int)
    parser.add_argument('--gpuid',          default=1, type=int)
    parser.add_argument('--prefix',         default='workstation', type=str)
    parser.add_argument('--posix',          default=None, type=str)
    parser.add_argument('--len_epoch_train', default=200, type=int)
    parser.add_argument('--len_epoch_valid', default=100, type=int)
    parser.add_argument('--accum_step',     default=1, type=int)
    parser.add_argument('--comment',        default='default-step', type=str)

    return parser


from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def dice_coeff_1(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
