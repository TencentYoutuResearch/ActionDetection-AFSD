import os
import multiprocessing as mp
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('thread_num', type=int)
parser.add_argument('--video_dir', type=str, default='datasets/activitynet/train_val_112')
parser.add_argument('--output_dir', type=str, default='datasets/activitynet/train_val_npy_112')
parser.add_argument('--max_frame_num', type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_dir = args.video_dir
output_dir = args.output_dir
max_frame_num = args.max_frame_num

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = sorted(os.listdir(video_dir))

def sub_processor(pid, files):
    for file in files[:]:
        file_name = os.path.splitext(file)[0]
        target_file = os.path.join(output_dir, file_name + '.npy')
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        imgs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs.append(frame[:, :, ::-1])
        if count != len(imgs):
            print('{} frame num is less'.format(file_name))
        imgs = np.stack(imgs)
        print(imgs.shape)
        if max_frame_num is not None:
            imgs = imgs[:max_frame_num]
        np.save(target_file, imgs)

processes = []
video_num = len(files)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = files[i * per_process_video_num:]
    else:
        sub_files = files[i * per_process_video_num: (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()