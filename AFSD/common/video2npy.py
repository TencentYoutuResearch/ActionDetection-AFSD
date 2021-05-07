import numpy as np
import cv2
import os
import pandas as pd
from AFSD.common.config import config
from AFSD.common.videotransforms import imresize


def print_videos_info(data_path):
    mp4_files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]
    for f in mp4_files:
        capture = cv2.VideoCapture(os.path.join(data_path, f))
        if not capture.isOpened():
            print('{} open failed!'.format(f))
        else:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            print('{}: fps={}, count={}, height={}, width={}'.format(
                f, fps, count, height, width
            ))


def video2npy(data_path, anno_path, save_path, sample_fps=10.0, resolution=112,
              export_video_info_path=None):
    df = pd.DataFrame(pd.read_csv(anno_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    video_infos = []
    for video_name in sorted(list(set(df['video'].values[:]))):
        capture = cv2.VideoCapture(os.path.join(data_path, video_name + '.mp4'))
        if not capture.isOpened():
            raise Exception('{} open failed!'.format(video_name))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        if fps <= 0:
            raise ValueError('{}: obtain wrong fps={}'.format(video_name, fps))
        if fps < sample_fps:
            raise ValueError('{}: sample fps {} is lower original fps {}'
                             .format(video_name, sample_fps, fps))

        step = fps / sample_fps
        cur_step = .0
        cur_count = 0
        save_count = 0
        res_frames = []
        while True:
            ret, frame = capture.read()
            if ret is False:
                break
            frame = np.array(frame)[:, :, ::-1]
            cur_count += 1
            cur_step += 1
            if cur_step >= step:
                cur_step -= step
                # save the frame
                target_img = imresize(frame, [resolution, resolution], 'bicubic')
                res_frames.append(target_img)
                save_count += 1

        if cur_count != int(count):
            raise ValueError('{}: total count {} is not equal to video count {}'.
                             format(video_name, cur_count, count))

        res_frames = np.stack(res_frames, 0)
        print('{}: result shape: {}'.format(video_name, res_frames.shape))

        video_infos.append([video_name, fps, sample_fps, count, save_count])
        # save to npy file
        np.save(os.path.join(save_path, video_name + '.npy'), res_frames)

    if export_video_info_path is not None:
        out_df = pd.DataFrame(video_infos,
                              columns=['video', 'fps', 'sample_fps', 'count', 'sample_count'])
        out_df.to_csv(export_video_info_path, index=False)


if __name__ == '__main__':
    video2npy(config['dataset']['training']['video_mp4_path'],
              config['dataset']['training']['video_anno_path'],
              config['dataset']['training']['video_data_path'],
              export_video_info_path=config['dataset']['training']['video_info_path'],
              sample_fps=10.0,
              resolution=112)

    video2npy(config['dataset']['testing']['video_mp4_path'],
              config['dataset']['testing']['video_anno_path'],
              config['dataset']['testing']['video_data_path'],
              export_video_info_path=config['dataset']['testing']['video_info_path'],
              sample_fps=10.0,
              resolution=112)
