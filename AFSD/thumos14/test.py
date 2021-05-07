import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map
from AFSD.thumos14.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config

num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
checkpoint_path = config['testing']['checkpoint_path']
json_name = config['testing']['output_json']
output_path = config['testing']['output_path']
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)
fusion = config['testing']['fusion']

# getting path for fusion
rgb_data_path = config['testing'].get('rgb_data_path',
                                      './datasets/thumos14/test_npy/')
flow_data_path = config['testing'].get('flow_data_path',
                                       './datasets/thumos14/test_flow_npy/')
rgb_checkpoint_path = config['testing'].get('rgb_checkpoint_path',
                                            './models/thumos14/checkpoint-15.ckpt')
flow_checkpoint_path = config['testing'].get('flow_checkpoint_path',
                                             './models/thumos14_flow/checkpoint-16.ckpt')

if __name__ == '__main__':
    video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
    originidx_to_idx, idx_to_class = get_class_index_map()

    npy_data_path = config['dataset']['testing']['video_data_path']
    if fusion:
        rgb_net = BDNet(in_channels=3, training=False)
        flow_net = BDNet(in_channels=2, training=False)
        rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
        flow_net.load_state_dict(torch.load(flow_checkpoint_path))
        rgb_net.eval().cuda()
        flow_net.eval().cuda()
        net = rgb_net
        npy_data_path = rgb_data_path
    else:
        net = BDNet(in_channels=config['model']['in_channels'],
                    training=False)

        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()

    if softmax_func:
        score_func = nn.Softmax(dim=-1)
    else:
        score_func = nn.Sigmoid()

    centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])

    result_dict = {}
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        sample_count = video_infos[video_name]['sample_count']
        sample_fps = video_infos[video_name]['sample_fps']
        if sample_count < clip_length:
            offsetlist = [0]
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]

        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data = centor_crop(data)
        data = torch.from_numpy(data)

        if fusion:
            flow_data = np.load(os.path.join(flow_data_path, video_name + '.npy'))
            flow_data = np.transpose(flow_data, [3, 0, 1, 2])
            flow_data = centor_crop(flow_data)
            flow_data = torch.from_numpy(flow_data)

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        # print(video_name)
        for offset in offsetlist:
            clip = data[:, offset: offset + clip_length]
            clip = clip.float()
            clip = (clip / 255.0) * 2.0 - 1.0
            if fusion:
                flow_clip = flow_data[:, offset: offset + clip_length]
                flow_clip = flow_clip.float()
                flow_clip = (flow_clip / 255.0) * 2.0 - 1.0
            # clip = torch.from_numpy(clip).float()
            if clip.size(1) < clip_length:
                tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                                   96, 96]).float()
                clip = torch.cat([clip, tmp], dim=1)
            clip = clip.unsqueeze(0).cuda()
            if fusion:
                if flow_clip.size(1) < clip_length:
                    tmp = torch.zeros([flow_clip.size(0), clip_length - flow_clip.size(1),
                                       96, 96]).float()
                    flow_clip = torch.cat([flow_clip, tmp], dim=1)
                flow_clip = flow_clip.unsqueeze(0).cuda()

            with torch.no_grad():
                output_dict = net(clip)
                if fusion:
                    flow_output_dict = flow_net(flow_clip)

            loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors']
            prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
            center = output_dict['center']
            if fusion:
                rgb_conf = conf[0]
                rgb_loc = loc[0]
                rgb_prop_loc = prop_loc[0]
                rgb_prop_conf = prop_conf[0]
                rgb_center = center[0]

                loc, conf, priors = flow_output_dict['loc'], flow_output_dict['conf'], \
                                    flow_output_dict['priors']
                prop_loc, prop_conf = flow_output_dict['prop_loc'], flow_output_dict['prop_conf']
                center = flow_output_dict['center']

                flow_conf = conf[0]
                flow_loc = loc[0]
                flow_prop_loc = prop_loc[0]
                flow_prop_conf = prop_conf[0]
                flow_center = center[0]

                loc = (rgb_loc + flow_loc) / 2.0
                prop_loc = (rgb_prop_loc + flow_prop_loc) / 2.0
                conf = (rgb_conf + flow_conf) / 2.0
                prop_conf = (rgb_prop_conf + flow_prop_conf) / 2.0
                center = (rgb_center + flow_center) / 2.0

            else:
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

            conf = score_func(conf)
            prop_conf = score_func(prop_conf)
            center = center.sigmoid()

            conf = (conf + prop_conf) / 2.0
            conf = conf * center
            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] > conf_thresh
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                # decode to original time
                # segments = (segments * clip_length + offset) / sample_fps
                segments = (segments + offset) / sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                output[cl].append(segments)
                # np.set_printoptions(precision=3, suppress=True)
                # print(idx_to_class[cl], tmp.detach().cpu().numpy())

        # print(output[1][0].size(), output[2][0].size())
        sum_count = 0
        for cl in range(1, num_classes):
            if len(output[cl]) == 0:
                continue
            tmp = torch.cat(output[cl], 0)
            tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
            res[cl, :count] = tmp
            sum_count += count

        sum_count = min(sum_count, top_k)
        flt = res.contiguous().view(-1, 3)
        flt = flt.view(num_classes, -1, 3)
        proposal_list = []
        for cl in range(1, num_classes):
            class_name = idx_to_class[cl]
            tmp = flt[cl].contiguous()
            tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
            if tmp.size(0) == 0:
                continue
            tmp = tmp.detach().cpu().numpy()
            for i in range(tmp.shape[0]):
                tmp_proposal = {}
                tmp_proposal['label'] = class_name
                tmp_proposal['score'] = float(tmp[i, 2])
                tmp_proposal['segment'] = [float(tmp[i, 0]),
                                           float(tmp[i, 1])]
                proposal_list.append(tmp_proposal)

        result_dict[video_name] = proposal_list

    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}

    with open(os.path.join(output_path, json_name), "w") as out:
        json.dump(output_dict, out)
