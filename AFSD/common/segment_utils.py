import torch
import numpy as np


def center_form(segments):
    """ convert (left, right) to (center, width) """
    return torch.cat([(segments[:, :1] - segments[:, 1:]) / 2.0,
                      segments[:, 1:] - segments[:, :1]], dim=1)


def point_form(segments):
    """ convert (centor, width) to (left, right) """
    return torch.cat([segments[:, :1] - segments[:, 1:] / 2.0,
                      segments[:, :1] + segments[:, 1:] / 2.0], dim=1)


def intersect(segment_a, segment_b):
    """
    for example, compute the max left between segment_a and segment_b.
    [A] -> [A, 1] -> [A, B]
    [B] -> [1, B] -> [A, B]
    """
    A = segment_a.size(0)
    B = segment_b.size(0)
    max_l = torch.max(segment_a[:, 0].unsqueeze(1).expand(A, B),
                      segment_b[:, 0].unsqueeze(0).expand(A, B))
    min_r = torch.min(segment_a[:, 1].unsqueeze(1).expand(A, B),
                      segment_b[:, 1].unsqueeze(0).expand(A, B))
    inter = torch.clamp(min_r - max_l, min=0)
    return inter


def jaccard(segment_a, segment_b):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    inter = intersect(segment_a, segment_b)
    length_a = (segment_a[:, 1] - segment_a[:, 0]).unsqueeze(1).expand_as(inter)
    length_b = (segment_b[:, 1] - segment_b[:, 0]).unsqueeze(0).expand_as(inter)
    union = length_a + length_b - inter
    return inter / union


def match_gt(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))
    # print(truths, point_form(priors))
    # print(overlaps)
    # [num_gt] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # [num_prior] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    # ensure each truth has one best prior
    best_truth_overlap.index_fill_(0, best_prior_idx, 2.0)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # [num_prior, 2]
    conf = labels[best_truth_idx]  # [num_prior]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def encode(matches, priors, variances):
    """
    :param matches: point form, shape: [num_priors, 2]
    :param priors: center form, shape: [num_priors, 2]
    :param variances: list of variances
    :return: encoded segments, shape: [num_priors, 2]
    """
    g_c = (matches[:, :1] + matches[:, 1:]) / 2.0 - priors[:, :1]
    g_c /= (variances[0] * priors[:, 1:])

    g_w = (matches[:, 1:] - matches[:, :1]) / priors[:, 1:]
    g_w = torch.log(g_w) / variances[1]

    return torch.cat([g_c, g_w], dim=1)  # [num_priors, 2]


def decode(loc, priors, variances):
    """
    :param loc: location predictions for loc layers, shape: [num_priors, 2]
    :param priors: center from, shape: [num_priors, 2]
    :param variances: list of variances
    :return: decoded segments, center form, shape: [num_priors, 2]
    """
    segments = torch.cat([
        priors[:, :1] + loc[:, :1] * priors[:, 1:] * variances[0],
        priors[:, 1:] * torch.exp(loc[:, 1:] * variances[1])], dim=1)
    return segments


def nms(segments, overlap=0.5, top_k=1000):
    left = segments[:, 0]
    right = segments[:, 1]
    scores = segments[:, 2]

    keep = scores.new_zeros(scores.size(0)).long()
    area = right - left
    v, idx = scores.sort(0)
    idx = idx[-top_k:]

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        l = torch.index_select(left, 0, idx)
        r = torch.index_select(right, 0, idx)
        l = torch.max(l, left[i])
        r = torch.min(r, right[i])
        # l = torch.clamp(l, max=left[i])
        # r = torch.clamp(r, min=right[i])
        inter = torch.clamp(r - l, min=0.0)

        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union

        idx = idx[IoU < overlap]
    return keep, count


def softnms_v2(segments, sigma=0.5, top_k=1000, score_threshold=0.001):
    segments = segments.cpu()
    tstart = segments[:, 0]
    tend = segments[:, 1]
    tscore = segments[:, 2]
    done_mask = tscore < -1  # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()

        undone_mask[idx] = False
        done_mask[idx] = True

        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou ** 2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    count = done_mask.sum()
    segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)
    return segments, count


def soft_nms(segments, overlap=0.3, sigma=0.5, top_k=1000):
    segments = segments.detach().cpu().numpy()
    tstart = segments[:, 0].tolist()
    tend = segments[:, 1].tolist()
    tscore = segments[:, 2].tolist()

    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) < top_k:
        max_score = max(tscore)
        if max_score < 0.001:
            break
        max_index = tscore.index(max_score)
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)

        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = np.maximum(tt2 - tt1, 0)
        duration = tend - tstart
        tmp_width = np.minimum(tmp_end - tmp_start, 1e-5)
        iou = intersection / (tmp_width + duration - intersection).astype(np.float)

        idxs = np.where(iou > overlap)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / sigma)

        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    count = len(rstart)
    rstart = np.array(rstart)
    rend = np.array(rend)
    rscore = np.array(rscore)
    segments = torch.from_numpy(np.stack([rstart, rend, rscore], axis=-1))
    return segments, count
