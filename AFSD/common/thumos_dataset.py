import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from AFSD.common import videotransforms
from AFSD.common.config import config
import random
import math


def get_class_index_map(class_info_path='thumos_annotations/Class Index_Detection.txt'):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4]
        }
    return video_infos


def get_video_anno(video_infos,
                   video_anno_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map()
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]['count']
        sample_count = video_infos[video_name]['sample_count']
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        if video_annos.get(video_name) is None:
            video_annos[video_name] = [[start_gt, end_gt, class_idx]]
        else:
            video_annos[video_name].append([start_gt, end_gt, class_idx])
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


def split_videos(video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 stride=config['dataset']['training']['clip_stride']):
    # video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    # video_annos = get_video_anno(video_infos,
    #                              config['dataset']['training']['video_anno_path'])
    training_list = []
    min_anno_dict = {}
    for video_name in video_annos.keys():
        min_anno = clip_length
        sample_count = video_infos[video_name]['sample_count']
        annos = video_annos[video_name]
        if sample_count <= clip_length:
            offsetlist = [0]
            min_anno_len = min([x[1] - x[0] for x in annos])
            if min_anno_len < min_anno:
                min_anno = min_anno_len
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]
        for offset in offsetlist:
            left, right = offset + 1, offset + clip_length
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa >= 0.5:
                    cur_annos.append([max(anno[0] - offset, 1),
                                      min(anno[1] - offset, clip_length),
                                      anno[2]])
            if len(cur_annos) > 0:
                min_anno_len = min([x[1] - x[0] for x in cur_annos])
                if min_anno_len < min_anno:
                    min_anno = min_anno_len
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                for anno in cur_annos:
                    s, e, id = anno
                    d = max((e - s) / 10.0, 2.0)
                    start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
                    start[start_s: start_e] = 1
                    end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
                    end[end_s: end_e] = 1
                training_list.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': cur_annos,
                    'start': start,
                    'end': end
                })
        min_anno_dict[video_name] = math.ceil(min_anno)
    return training_list, min_anno_dict


def load_video_data(video_infos, npy_data_path):
    data_dict = {}
    print('loading video frame data ...')
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data_dict[video_name] = data
    return data_dict


class THUMOS_Dataset(Dataset):
    def __init__(self, data_dict,
                 video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 crop_size=config['dataset']['training']['crop_size'],
                 stride=config['dataset']['training']['clip_stride'],
                 rgb_norm=True,
                 training=True,
                 origin_ratio=0.5):
        self.training_list, self.th = split_videos(
            video_infos,
            video_annos,
            clip_length,
            stride
        )
        # np.random.shuffle(self.training_list)
        self.data_dict = data_dict
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)
        self.rgb_norm = rgb_norm
        self.training = training

        self.origin_ratio = origin_ratio

    def __len__(self):
        return len(self.training_list)

    def get_bg(self, annos, min_action):
        annos = [[anno[0], anno[1]] for anno in annos]
        times = []
        for anno in annos:
            times.extend(anno)
        times.extend([0, self.clip_length - 1])
        times.sort()
        regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
        regions = list(filter(
            lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action, regions))
        # regions = list(filter(lambda x:x not in annos, regions))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]

    def augment_(self, input, annos, th):
        '''
        input: (c, t, h, w)
        target: (N, 3)
        '''
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
            # gt = random.choice(annos)
        except IndexError:
            return input, annos, False
        gt_len = gt[1] - gt[0]
        region = range(math.floor(th), math.ceil(gt_len - th))
        t = random.choice(region) + math.ceil(gt[0])
        l_len = math.ceil(t - gt[0])
        r_len = math.ceil(gt[1] - t)
        try:
            bg = self.get_bg(annos, th)
        except IndexError:
            return input, annos, False
        start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
        end_idx = start_idx + th

        new_input = input.clone()
        # annos.remove(gt)
        if gt[1] < start_idx:
            new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
            new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]

            new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
            # new_annos = [[t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t+th-math.ceil(th/5), t+th+math.ceil(th/5)],
            #            [t+1, t+th-1]]

        else:
            new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
            new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]

            new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
            # new_annos = [[t-th-math.ceil(th/5), t-th+math.ceil(th/5)],
            #            [t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t-th+1, t-1]]

        return new_input, new_annos, True

    def augment(self, input, annos, th, max_iter=10):
        flag = True
        i = 0
        while flag and i < max_iter:
            new_input, new_annos, flag = self.augment_(input, annos, th)
            i += 1
        return new_input, new_annos, flag

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_data = self.data_dict[sample_info['video_name']]
        offset = sample_info['offset']
        annos = sample_info['annos']
        th = self.th[sample_info['video_name']]

        input_data = video_data[:, offset: offset + self.clip_length]
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            # padding t to clip_length
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)

        # random crop and flip
        if self.training:
            input_data = self.random_flip(self.random_crop(input_data))
        else:
            input_data = self.center_crop(input_data)

        # import pdb;pdb.set_trace()
        input_data = torch.from_numpy(input_data).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        ssl_input_data, ssl_annos, flag = self.augment(input_data, annos, th, 1)
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)
        ssl_target = np.stack(ssl_annos, 0)

        scores = np.stack([
            sample_info['start'],
            sample_info['end']
        ], axis=0)
        scores = torch.from_numpy(scores.copy()).float()

        return input_data, target, scores, ssl_input_data, ssl_target, flag


def detection_collate(batch):
    targets = []
    clips = []
    scores = []

    ssl_targets = []
    ssl_clips = []
    flags = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        scores.append(sample[2])

        ssl_clips.append(sample[3])
        ssl_targets.append(torch.FloatTensor(sample[4]))
        flags.append(sample[5])
    return torch.stack(clips, 0), targets, torch.stack(scores, 0), \
           torch.stack(ssl_clips, 0), ssl_targets, flags
