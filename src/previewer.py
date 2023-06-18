import cv2
import math
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
from multiprocess import Pool

from openpose import util

def previewbatch(imgbatch, real, fake, box=None, n=16):

    try:
        fake = [_.detach().float().cpu().numpy() for _ in fake]
    except AttributeError:
        pass
    
    ret = list()
    pcks = np.zeros((1, 20))
    
    n_ = min(n, len(real[0]))
    # 
    results = list()
    # for i in range(min(n, len(imgbatch))):
    def subprocess(i):
        canvas = imgbatch[i] * 255 + 127
        canvas = np.array(np.ascontiguousarray(canvas.squeeze().transpose()), dtype=np.uint8)
        candidate_r, subset_r = _preview(canvas, real[1][i], real[0][i])
        canvas = util.draw_bodypose(np.ascontiguousarray(canvas.transpose(1, 2, 0)), candidate_r, subset_r)
        ret.append(canvas)

        canvas = imgbatch[i] * 255 + 127
        canvas = np.ascontiguousarray(canvas.squeeze().transpose())
        candidate_f, subset_f = _preview(canvas, fake[1][i], fake[0][i])
        canvas = util.draw_bodypose(np.ascontiguousarray(canvas.transpose(1, 2, 0)), candidate_f, subset_f)
        ret.append(canvas)
        
        # results.append([ret, candidate_f, candidate_r, subset_f, subset_r])
        
        return ret, candidate_f, candidate_r, subset_f, subset_r
        
    pool = Pool()
    results = pool.map(subprocess, range(n_))
    pool.close()
    
    ret = list()
    for i in results:
        ret.extend(i[0])

    for i in range(n_):
        if box is not None:
            pcks += pck(results[i][4], results[i][3], results[i][2], results[i][1], box=None)

    return np.stack(ret)[:, :, :, [2, 1, 0]], pcks / n if box is not None else None, min(n, len(imgbatch))


def pck(subset_real, subset_fake, candidate_real, candidate_fake, box=None, orisize=512, bars=[_ / 20 for _ in range(20)]):

    num_gt = subset_real[0][-1]
    
    dis = [1.01]
    normer = 1
    if len(subset_fake) > 0:
        dis = list()
        gtxlist = list()
        gtylist = list()
    
        for i in range(17):
            if (subset_real[0][i] != -1) & (subset_fake[0][i] != -1):
                # gt = candidate_real[int(subset_real[0][i])][:2] / 368
                # pd = candidate_fake[int(subset_fake[0][i])][:2] / 368
                
                gtx = candidate_real[int(subset_real[0][i])][0] / 960
                gty = candidate_real[int(subset_real[0][i])][1] / 540

                pdx = candidate_fake[int(subset_fake[0][i])][0] / 960
                pdy = candidate_fake[int(subset_fake[0][i])][1] / 540
                
                dis.append(((gtx - pdx) ** 2 + (gty - pdy) ** 2) ** 2)
                
                gtxlist.append(gtx)
                gtylist.append(gty)
                
                # box_normed = box[0] / 512
                # normer = np.sqrt((box_normed[2] - box_normed[0]) ** 2 + (box_normed[3] - box_normed[1]) ** 2)
                # dis.append(np.sqrt(np.sum((gt - pd) ** 2)) / normer)
            if len(gtxlist) > 0:    
                normer = ((max(gtxlist) - min(gtxlist)) ** 2 + (max(gtylist) - min(gtylist)) ** 2) ** 0.5
            else:
                normer = 1
                    
    scores = np.sum(np.array([dis]) / normer < (1 - np.array([bars])).T, axis=1) / num_gt
    return scores.squeeze()


def _preview(oriImg, Mconv7_stage6_L1, Mconv7_stage6_L2):
    oriImg = np.zeros((3, 960, 540))
    scale_search = [1.]
    scale = 1.
    # boxsize = 368
    boxsize = 512
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.005
    
    x = 960
    y = 540
    
    Mconv7_stage6_L1 = np.array(Mconv7_stage6_L1, dtype=float)
    Mconv7_stage6_L2 = np.array(Mconv7_stage6_L2, dtype=float)

    oriImg = np.ascontiguousarray(oriImg).transpose(1, 2, 0)

    multiplier = [x * boxsize / oriImg.shape[1] for x in scale_search]
    heatmap_avg = np.zeros((y, x, 19))
    paf_avg = np.zeros((y, x, 38))
    # heatmap_avg = np.zeros((oriImg.shape[1], oriImg.shape[0], 19))
    # paf_avg = np.zeros((oriImg.shape[1], oriImg.shape[0], 38))

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
    
    heatmap = np.squeeze(Mconv7_stage6_L2).transpose(1, 2, 0)  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[0], oriImg.shape[1]), interpolation=cv2.INTER_CUBIC)

    paf = np.squeeze(Mconv7_stage6_L1).transpose(1, 2, 0)  # output 0 is PAFs
    paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[0], oriImg.shape[1]), interpolation=cv2.INTER_CUBIC)

    heatmap_avg += heatmap_avg + heatmap / len(multiplier)
    paf_avg += + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                [55, 56], [37, 38], [45, 46]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = min(len(candA), 10)
        nB = min(len(candB), 10)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                        for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                        for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(min(len(connection_candidate), 300)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:100, 0]
            partBs = connection_all[k][:100, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
    # candidate: x, y, score, id
    return candidate, subset
