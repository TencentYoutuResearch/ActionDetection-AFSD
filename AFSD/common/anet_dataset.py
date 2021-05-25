import json
import torch
from torch.utils.data import Dataset
from AFSD.common import videotransforms
import os
import numpy as np
import math
import random


def load_json(file):
    """
    :param file: json file path
    :return: data of json
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


def get_video_info(video_info_path, subset='training'):
    json_data = load_json(video_info_path)
    video_info = {}
    video_list = list(json_data.keys())
    for video_name in video_list:
        tmp = json_data[video_name]
        if tmp['subset'] == subset:
            video_info[video_name] = tmp
    return video_info


def split_videos(video_info, clip_length, stride, binary_class=False):
    training_list = []
    min_anno_dict = {}
    for video_name in list(video_info.keys())[:]:
        frame_num = min(video_info[video_name]['frame_num'], clip_length)
        annos = []
        min_anno = clip_length
        for anno in video_info[video_name]['annotations']:
            if binary_class:
                anno['label_id'] = 1 if anno['label_id'] > 0 else 0
            if anno['end_frame'] <= anno['start_frame']:
                continue
            annos.append([
                anno['start_frame'],
                anno['end_frame'],
                anno['label_id']
            ])
        if len(annos) == 0:
            continue

        offsetlist = [0]

        for offset in offsetlist:
            cur_annos = []
            save_offset = True
            for anno in annos:
                cur_annos.append([anno[0], anno[1], anno[2]])
            if len(cur_annos) > 0:
                min_anno_len = min([x[1] - x[0] for x in cur_annos])
                if min_anno_len < min_anno:
                    min_anno = min_anno_len
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                action = np.zeros([clip_length])
                for anno in cur_annos:
                    s, e, id = anno
                    d = max((e - s) / 10.0, 2.0)
                    act_s = np.clip(int(round(s)), 0, clip_length - 1)
                    act_e = np.clip(int(round(e)), 0, clip_length - 1) + 1
                    action[act_s: act_e] = id
                    start_s = np.clip(int(round(s - d / 2)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2)), 0, clip_length - 1) + 1
                    start[start_s: start_e] = id
                    end_s = np.clip(int(round(e - d / 2)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2)), 0, clip_length - 1) + 1
                    end[end_s: end_e] = id

                training_list.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': cur_annos,
                    'frame_num': frame_num,
                    'start': start,
                    'end': end,
                    'action': action
                })
        min_anno_dict[video_name] = math.floor(min_anno)
    return training_list, min_anno_dict


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


class ANET_Dataset(Dataset):
    def __init__(self,
                 video_info_path,
                 video_dir,
                 clip_length,
                 crop_size,
                 stride,
                 channels=3,
                 rgb_norm=True,
                 training=True,
                 binary_class=False):
        self.training = training
        subset = 'training' if training else 'validation'
        video_info = get_video_info(video_info_path, subset)
        self.training_list, self.th = split_videos(video_info, clip_length, stride, binary_class)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.rgb_norm = rgb_norm
        self.video_dir = video_dir
        self.channels = channels

        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)

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
        regions = list(
            filter(lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action,
                   regions))
        # regions = list(filter(lambda x:x not in annos, regions))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]

    def augment_(self, input, annos, th):
        '''
        input: (c, t, h, w)
        target: (N, 3)
        '''
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] >= 2 * th, annos)))
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
        try:
            if gt[1] < start_idx:
                new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
                new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]

                new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]

            else:
                new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
                new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]

                new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
        except RuntimeError:
            return input, annos, False

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
        video_name = sample_info['video_name']
        offset = sample_info['offset']
        annos = sample_info['annos']
        frame_num = sample_info['frame_num']
        th = int(self.th[sample_info['video_name']] / 4)
        data = np.load(os.path.join(self.video_dir, video_name + '.npy'))
        start = offset
        end = min(offset + self.clip_length, frame_num)
        frames = data[start: end]
        frames = np.transpose(frames, [3, 0, 1, 2]).astype(np.float)

        c, t, h, w = frames.shape
        if t < self.clip_length:
            pad_t = self.clip_length - t
            zero_clip = np.ones([c, pad_t, h, w], dtype=frames.dtype) * 127.5
            frames = np.concatenate([frames, zero_clip], 1)

        # random crop and flip
        if self.training:
            frames = self.random_flip(self.random_crop(frames))
        else:
            frames = self.center_crop(frames)

        input_data = torch.from_numpy(frames.copy()).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        ssl_input_data, ssl_annos, flag = self.augment(input_data, annos, th, 1)
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)
        ssl_target = np.stack(ssl_annos, 0)

        scores = np.stack([
            sample_info['action'],
            sample_info['start'],
            sample_info['end']
        ], axis=0)
        scores = torch.from_numpy(scores.copy()).float()

        return input_data, target, scores, ssl_input_data, ssl_target, flag
