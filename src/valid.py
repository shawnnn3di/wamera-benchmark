import argparse
import numpy as np
import pickle as pk
import tensorboardX
import torch
import tqdm

import dataparse, includeargs, modeling, previewer, utils, wtils


if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser()
    parser = includeargs.include_args(parser)
    args = parser.parse_args()
    
    # gpu setting
    device = 'cuda:%d' % args.gpuid
    
    # data loading
    _, valid_loader = dataparse.build_loader(args, validonly=True)
    
    # models
    model, optimizer, scheduler, forward = modeling.build_model(args, device, checkpoint='/home/lscsc/caizhijie/0420-wamera-benchmark/pipeline_0518test_60.checkpoint')
    
    # dump
    dumplist = list()
    dumplist.append('i, loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img, box')
    
    loss_recorder = utils.recorder_vector(keys=['loss', 'loss_sm', 'loss_jhm', 'loss_paf'])
    pcks_recorder = utils.recorder_vector(keys=['%.2f' % (float(_) / 20) for _ in range(20)])
    
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader))
        lenpbar = len(pbar)
        for i, batch in pbar:
            
            loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img, box = forward(batch, device, model, args.half)
            loss = loss_sm + loss_jhm + loss_paf
            loss_recorder.update(np.array(utils.cpunumpy([loss, loss_sm, loss_jhm, loss_paf])), sm.shape[0])
            img_batch, pcks, n = previewer.previewbatch(img, utils.cpunumpy([y_jhm, y_paf]), (jhm, paf), box)
            pcks_recorder.update(pcks, n)
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % ('valid', 1, 1, i, lenpbar, loss))
            
            dumplist.append([i] + utils.cpunumpy([loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf]) + [img_batch, box])
            
    # dump to
    pk.dump(dumplist, open('validdumps.pk', 'wb'))
            