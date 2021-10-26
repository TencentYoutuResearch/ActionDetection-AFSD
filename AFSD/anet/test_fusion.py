import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.anet_dataset import get_video_info, load_json
from AFSD.anet.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config

import multiprocessing as mp
import threading


num_classes = 2
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
crop_size = config['dataset']['testing']['crop_size']
rgb_checkpoint_path = 'models/anet/checkpoint-10.ckpt'
flow_checkpoint_path = 'models/anet_flow/checkpoint-6.ckpt'
json_name = config['testing']['output_json']
output_path = config['testing']['output_path']
ngpu = config['ngpu']
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)

thread_num = ngpu
global result_dict
result_dict = mp.Manager().dict()

processes = []
lock = threading.Lock()

video_infos = get_video_info(config['dataset']['testing']['video_info_path'],
                             subset='validation')
rgb_mp4_data_path = 'datasets/activitynet/train_val_npy_112'
flow_mp4_data_path = 'datasets/activitynet/flow/train_val_npy_112'

if softmax_func:
    score_func = nn.Softmax(dim=-1)
else:
    score_func = nn.Sigmoid()

centor_crop = videotransforms.CenterCrop(crop_size)

video_list = list(video_infos.keys())
video_num = len(video_list)
per_thread_video_num = video_num // thread_num

cuhk_data = load_json('cuhk-val/cuhk_val_simp_share.json')
cuhk_data_score = cuhk_data["results"]
cuhk_data_action = cuhk_data["class"]

def sub_processor(lock, pid, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)
    rgb_net = BDNet(in_channels=3, training=False)
    flow_net = BDNet(in_channels=2, training=False)
    rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
    flow_net.load_state_dict(torch.load(flow_checkpoint_path))
    rgb_net.eval().cuda()
    flow_net.eval().cuda()

    for video_name in video_list:
        cuhk_score = cuhk_data_score[video_name[2:]]
        cuhk_class_1 = cuhk_data_action[np.argmax(cuhk_score)]
        cuhk_score_1 = max(cuhk_score)

        sample_count = video_infos[video_name]['frame_num']
        sample_fps = video_infos[video_name]['fps']
        duration = video_infos[video_name]['duration']

        offsetlist = [0]

        data = np.load(os.path.join(rgb_mp4_data_path, video_name + '.npy'))
        frames = data
        frames = np.transpose(frames, [3, 0, 1, 2])
        data = centor_crop(frames)
        data = torch.from_numpy(data.copy())
        rgb_data = data

        data = np.load(os.path.join(flow_mp4_data_path, video_name + '.npy'))
        frames = data
        frames = np.transpose(frames, [3, 0, 1, 2])
        data = centor_crop(frames)
        data = torch.from_numpy(data.copy())
        flow_data = data

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        for offset in offsetlist:
            rgb_clip = rgb_data[:, offset: offset + clip_length]
            rgb_clip = rgb_clip.float()

            flow_clip = flow_data[:, offset: offset + clip_length]
            flow_clip = flow_clip.float()

            if rgb_clip.size(1) < clip_length:
                rgb_tmp = torch.ones(
                    [rgb_clip.size(0), clip_length - rgb_clip.size(1), crop_size, crop_size]).float() * 127.5
                flow_tmp = torch.ones(
                    [flow_clip.size(0), clip_length - flow_clip.size(1), crop_size, crop_size]).float() * 127.5
                rgb_clip = torch.cat([rgb_clip, rgb_tmp], dim=1)
                flow_clip = torch.cat([flow_clip, flow_tmp], dim=1)
            rgb_clip = rgb_clip.unsqueeze(0).cuda()
            flow_clip = flow_clip.unsqueeze(0).cuda()
            rgb_clip = (rgb_clip / 255.0) * 2.0 - 1.0
            flow_clip = (flow_clip / 255.0) * 2.0 - 1.0

            with torch.no_grad():
                rgb_output_dict = rgb_net(rgb_clip)
                flow_output_dict = flow_net(flow_clip)

            loc, conf, priors = rgb_output_dict['loc'], rgb_output_dict['conf'], \
                                rgb_output_dict['priors'][0]
            prop_loc, prop_conf = rgb_output_dict['prop_loc'], rgb_output_dict['prop_conf']
            center = rgb_output_dict['center']

            loc = loc[0]
            conf = conf[0]
            prop_loc = prop_loc[0]
            prop_conf = prop_conf[0]
            center = center[0]

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                 priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length)
            rgb_segments = decoded_segments

            rgb_loc = loc
            rgb_prop_conf = prop_conf
            rgb_prop_loc = prop_loc
            rgb_conf = conf
            rgb_center = center

            loc, conf, priors = flow_output_dict['loc'], flow_output_dict['conf'], \
                                flow_output_dict['priors'][0]
            prop_loc, prop_conf = flow_output_dict['prop_loc'], flow_output_dict['prop_conf']
            center = flow_output_dict['center']

            loc = loc[0]
            conf = conf[0]
            prop_loc = prop_loc[0]
            prop_conf = prop_conf[0]
            center = center[0]

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                 priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length)
            flow_segments = decoded_segments

            flow_loc = loc
            flow_prop_loc = prop_loc
            flow_prop_conf = prop_conf
            flow_conf = conf
            flow_center = center

            loc = (rgb_loc + flow_loc) / 2.0
            prop_loc = (rgb_prop_loc + flow_prop_loc) / 2.0
            conf = (rgb_conf + flow_conf) / 2.0
            prop_conf = (rgb_prop_conf + flow_prop_conf) / 2.0
            center = (rgb_center + flow_center) / 2.0

            decoded_segments = torch.sqrt(rgb_segments * flow_segments)

            conf = score_func(conf)
            prop_conf = score_func(prop_conf)
            conf = (conf + prop_conf) / 2.0
            center = center.sigmoid()
            conf = conf * center

            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] > 0
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                segments = (segments + offset) / sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                output[cl].append(segments)

        sum_count = 0
        for cl in range(1, num_classes):
            if len(output[cl]) == 0:
                continue
            tmp = torch.cat(output[cl], 0)
            tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k, score_threshold=1e-18)
            res[cl, :count] = tmp
            sum_count += count

        flt = res.contiguous().view(-1, 3)
        flt = flt.view(num_classes, -1, 3)
        proposal_list = []
        for cl in range(1, num_classes):
            class_name = cuhk_class_1
            tmp = flt[cl].contiguous()
            tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
            if tmp.size(0) == 0:
                continue
            tmp = tmp.detach().cpu().numpy()
            for i in range(tmp.shape[0]):
                tmp_proposal = {}
                start_time = max(0, float(tmp[i, 0]))
                end_time = min(duration, float(tmp[i, 1]))
                if end_time <= start_time:
                    continue

                tmp_proposal['label'] = class_name
                tmp_proposal['score'] = float(tmp[i, 2]) * cuhk_score_1
                tmp_proposal['segment'] = [start_time, end_time]
                proposal_list.append(tmp_proposal)

        result_dict[video_name[2:]] = proposal_list
        with lock:
            progress.update(1)
    with lock:
        progress.close()

for i in range(thread_num):
    if i == thread_num - 1:
        sub_video_list = video_list[i * per_thread_video_num:]
    else:
        sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {}}

with open(os.path.join(output_path, json_name), "w") as out:
    json.dump(output_dict, out)
