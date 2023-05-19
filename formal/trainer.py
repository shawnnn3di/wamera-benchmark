import argparse
import numpy as np
import pickle as pk
import tensorboardX
import torch
import tqdm

import dataparse, includeargs, modeling, previewer, utils, wtils


def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser()
    parser = includeargs.include_args(parser)
    args = parser.parse_args()
    
    # gpu setting
    device = 'cuda:%d' % args.gpuid
    
    # tensorboard init
    writer = tensorboardX.SummaryWriter('./tensorboard/%s' % args.comment, 'trainer')
    
    # dump options
    if args.dump_loss_gap > 0:
        dump = dict()
    
    # colorset for preview
    n_color = 19
    colors, hls = utils.ncolors(n_color)
    colors[n_color - 1] = [0, 0, 0]
    
    # data loading
    train_loader, valid_loader = dataparse.build_loader(args)
    
    # models
    model, optimizer, scheduler, forward = modeling.build_model(args, device)
        
    # iterate
    for j in range(args.num_epoch):
        
        # valid
        loss_recorder = utils.recorder_vector(keys=['loss', 'loss_sm', 'loss_jhm', 'loss_paf'])
        if j % args.valid_gap == 0:
            pcks_recorder = utils.recorder_vector(keys=['%.2f' % (float(_) / 20) for _ in range(20)])
            
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader))
                lenpbar = len(pbar)
                for i, batch in pbar:
                    
                    loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img, box = forward(batch, device, model, args.half)
                    loss = loss_sm + loss_jhm + loss_paf
                    loss_recorder.update(np.array(utils.cpunumpy([loss, loss_sm, loss_jhm, loss_paf])), sm.shape[0])
                    
                    if i % args.preview_gap == 0:
                        wtils.writer_preview(writer, colors, jhm, sm, y_jhm, y_sm, j, 'valid')
                        img_batch, pcks = previewer.previewbatch(img, utils.cpunumpy([y_jhm, y_paf]), (jhm, paf), box)
                        pcks_recorder.update(pcks)
                    pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % ('valid', j, args.num_epoch, i, lenpbar, loss))
                
            wtils.writer_epochwrap(writer, loss_recorder, pcks_recorder, j, 'valid')
            if (args.dump_loss_gap > 0) and (j % args.dump_loss_gap == 0):
                dump.update({j: [loss_recorder.avg(), pcks_recorder.avg()]})
            
        # train
        loss_recorder = utils.recorder_vector(keys=['loss', 'loss_sm', 'loss_jhm', 'loss_paf'])
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        lenpbar = len(pbar)
        for i, batch in pbar:
            loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, img, box = forward(batch, device, model, args.half)
            loss = loss_sm + loss_jhm + loss_paf
            loss_recorder.update(np.array(utils.cpunumpy([loss, loss_sm, loss_jhm, loss_paf])), sm.shape[0])
            
            backward(optimizer, loss)
            
            if i % args.preview_gap == 0:
                wtils.writer_preview(writer, colors, jhm, sm, y_jhm, y_sm, j)
            pbar.set_description('%s, epoch: %d/%d, batch: %d/%d, loss: %.4f' % ('train', j, args.num_epoch, i, lenpbar, loss))
        
        wtils.writer_epochwrap(writer, loss_recorder, pcks_recorder, j, 'train')

        # checkpoint
        if j % args.checkpoint_gap == 0:
            utils.checkpoint(j, model, args)
            
    # final dump
    if args.dump_loss_gap > 0:
        pk.dump(dump, open('./tensorboard/%s' % args.comment + 'dump.pk', 'wb'))
    