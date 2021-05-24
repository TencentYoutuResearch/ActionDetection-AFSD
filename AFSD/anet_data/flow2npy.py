import cv2
import os
import numpy as np
import json
import glob
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('thread_num', type=int)
parser.add_argument('--video_info_path', type=str,
                    default='anet_annotations/video_info_train_val.json')
parser.add_argument('--flow_frame_path', type=str,
                    default='datasets/activitynet/flow/frame_train_val_112')
parser.add_argument('--flow_npy_path', type=str,
                    default='datasets/activitynet/flow/train_val_npy_112')
parser.add_argument('--max_frame_num', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_info_path = args.video_info_path
flow_frame_path = args.flow_frame_path
flow_npy_path = args.flow_npy_path
max_frame_num = args.max_frame_num


def load_json(file):
    """
    :param file: json file path
    :return: data of json
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data


if not os.path.exists(flow_npy_path):
    os.makedirs(flow_npy_path)

json_data = load_json(video_info_path)

video_list = sorted(list(json_data.keys()))


def sub_processor(pid, video_list):
    for video_name in video_list:
        tmp = []
        print(video_name)
        flow_x_files = sorted(glob.glob(os.path.join(flow_frame_path, video_name, 'flow_x_*.jpg')))
        flow_y_files = sorted(glob.glob(os.path.join(flow_frame_path, video_name, 'flow_y_*.jpg')))
        assert len(flow_x_files) > 0
        assert len(flow_x_files) == len(flow_y_files)

        frame_num = json_data[video_name]['frame_num']
        fps = json_data[video_name]['fps']

        output_file = os.path.join(flow_npy_path, video_name + '.npy')

        while len(flow_x_files) < frame_num:
            flow_x_files.append(flow_x_files[-1])
            flow_y_files.append(flow_y_files[-1])
        for flow_x, flow_y in zip(flow_x_files, flow_y_files):
            flow_x = cv2.imread(flow_x)[:, :, 0]
            flow_y = cv2.imread(flow_y)[:, :, 0]
            img = np.stack([flow_x, flow_y], -1)
            tmp.append(img)

        tmp = np.stack(tmp, 0)
        if max_frame_num is not None:
            tmp = tmp[:max_frame_num]
        np.save(output_file, tmp)


processes = []
video_num = len(video_list)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = video_list[i * per_process_video_num:]
    else:
        sub_files = video_list[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
