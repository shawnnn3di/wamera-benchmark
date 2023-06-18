import numpy as np


def writer_preview(writer, colors, jhm, sm, y_jhm, y_sm, j, mode='train'):
    arr = np.clip(np.matmul(jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
    
    # arr = np.clip(np.matmul(np.clip(np.stack(np.array_split(jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), 4, -1)), 0, 1), np.array(colors[:19])) / 19, 0, 255).reshape((-1, 96, 54, 3))
    writer.add_images('%s_jhm_pd' % mode, arr, j, dataformats='NHWC')
    writer.add_images('%s_sm_pd' % mode, sm.unsqueeze(-1), j, dataformats='NHWC')
    
    # arr = np.clip(np.matmul(np.clip(np.stack(y_jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), 4, -1)), np.array(colors[:19])) / 19, 0, 255).reshape((-1, 96, 54, 3))
    
    arr = np.clip(np.matmul(y_jhm.cpu().detach().numpy().transpose(0, 3, 2, 1), np.array(colors)[:19, :]), 0, 255)
    writer.add_images('%s_jhm_gt' % mode, arr, j, dataformats='NHWC')
    # writer.add_images('%s_sm_gt' % mode, y_sm.unsqueeze(-1), j, dataformats='NHWC')
    
    
def writer_epochwrap(writer, loss_recorder, pcks_recorder, j, mode='train'):
    writer.add_scalars('%s_loss' % mode, loss_recorder.avg(), global_step=j)
    writer.add_scalars('%s_pcks' % mode, pcks_recorder.avg(), global_step=j)