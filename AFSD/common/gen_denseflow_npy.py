import numpy as np
import cv2
import os
import tqdm
import glob
from AFSD.common.config import config
from AFSD.common.thumos_dataset import get_video_info
from AFSD.common.videotransforms import imresize

"""
Following I3D data preprocessing, for the flow stream, we convert the videos to grayscale,
and pixel values are truncated to the range [-20, 20], then rescaled between -1 and 1. 
We only use the first two output dimensions, and apply the same cropping as for RGB. 
"""


def gen_flow_image_from_video(video_info_path, video_mp4_path, output_dir):
    video_info = get_video_info(video_info_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in list(video_info.keys()):
        mp4_path = os.path.join(video_mp4_path, video_name + '.mp4')
        os.system('denseflow {} -b=20 -a=tvl1 -s=1 -o={} -v'.format(mp4_path,
                                                                    output_dir))


def gen_flow_npy_with_sample(video_info_path, video_flow_img_path, output_dir, new_size):
    video_info = get_video_info(video_info_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in list(video_info.keys()):
        npy_path = os.path.join(output_dir, video_name + '.npy')
        if os.path.exists(npy_path):
            print('{} is existed.'.format(npy_path))
            continue
        fps = video_info[video_name]['fps']
        sample_fps = video_info[video_name]['sample_fps']
        sample_count = video_info[video_name]['sample_count']

        step = fps / sample_fps
        flow_x_imgs = sorted(glob.glob(
            os.path.join(video_flow_img_path, video_name, 'flow_x_*.jpg')))
        flow_y_imgs = sorted(glob.glob(
            os.path.join(video_flow_img_path, video_name, 'flow_y_*.jpg')))
        cur_step = .0

        flows = []
        for flow_x_img, flow_y_img in zip(flow_x_imgs, flow_y_imgs):
            cur_step += 1
            if cur_step >= step:
                cur_step -= step
                flow_x = cv2.imread(flow_x_img)
                flow_x = imresize(flow_x, new_size, interp='bicubic')[:, :, 0]
                flow_y = cv2.imread(flow_y_img)
                flow_y = imresize(flow_y, new_size, interp='bicubic')[:, :, 0]
                flows.append(np.stack([flow_x, flow_y], axis=-1))

        while len(flows) < sample_count:
            flows.append(flows[-1])
        # print(len(flows), sample_count)
        assert len(flows) == sample_count
        flows = np.stack(flows, axis=0)
        assert flows.dtype == np.uint8
        # print(flows.shape)
        np.save(npy_path, flows)


def gen_flow_image(video_info_path, video_data_path, output_dir):
    video_info = get_video_info(video_info_path)
    for video_name in list(video_info.keys())[:]:
        npy_path = os.path.join(video_data_path, video_name + '.npy')

        if not os.path.exists(video_name):
            os.makedirs(video_name)

        imgs = np.load(npy_path)
        imgs = imgs[:, :, :, ::-1]  # convert RGB to BGR
        # gray_imgs = []
        for i in range(imgs.shape[0]):
            im = imgs[i]
            cv2.imwrite(os.path.join(video_name, '%05d.jpg' % (i + 1)), im)
        os.system('denseflow {} -b=20 -a=tvl1 -s=1 -if -v -o={}'.format(video_name, output_dir))
        os.system('rm {} -r'.format(video_name))


def gen_flow_npy(video_info_path, video_flow_img_path, output_dir):
    video_info = get_video_info(video_info_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for video_name in tqdm.tqdm(list(video_info.keys())):
        img_path = os.path.join(video_flow_img_path, video_name)
        count = video_info[video_name]['sample_count']
        npy_path = os.path.join(output_dir, video_name + '.npy')
        flows = []
        for i in range(count - 1):
            flow_x = cv2.imread(os.path.join(img_path, 'flow_x_%05d.jpg' % i))[:, :, 0]
            flow_y = cv2.imread(os.path.join(img_path, 'flow_y_%05d.jpg' % i))[:, :, 0]
            flow = np.stack([flow_x, flow_y], axis=-1)
            flows.append(flow)
        flows.append(flows[-1])
        flows = np.stack(flows, axis=0)
        # print(flows.shape, flows.dtype)
        np.save(npy_path, flows)


if __name__ == '__main__':
    gen_flow_image(config['dataset']['training']['video_info_path'],
                   config['dataset']['training']['video_data_path'],
                   './datasets/thumos14/validation_flows')

    gen_flow_image(config['dataset']['testing']['video_info_path'],
                   config['dataset']['testing']['video_data_path'],
                   './datasets/thumos14/test_flows')

    gen_flow_npy(config['dataset']['training']['video_info_path'],
                 './datasets/thumos14/validation_flows',
                 './datasets/thumos14/validation_flow_npy')

    gen_flow_npy(config['dataset']['testing']['video_info_path'],
                 './datasets/thumos14/test_flows',
                 './datasets/thumos14/test_flow_npy')
