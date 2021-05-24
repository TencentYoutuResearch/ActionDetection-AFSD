import os
import json
from AFSD.anet_data.class_map import class_to_id
import cv2

origin_video_info_path = 'anet_annotations/video_info_19993.json'
new_video_info_path = 'anet_annotations/video_info_train_val.json'
video_dir = 'datasets/activitynet/train_val_112'

def load_json(file):
    """
    :param file: json file path
    :return: data of json
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data

new_video_info = {}
json_data = load_json(origin_video_info_path)
video_list = list(json_data.keys())
for video_name in video_list:
    subset = json_data[video_name]['subset']
    if subset == 'testing':
        continue
    tmp_info = {}
    tmp_info['subset'] = subset
    tmp_info['duration'] = json_data[video_name]['duration']
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name + '.mp4'))
    if not cap.isOpened():
        print('error:', video_name)
        exit()
    tmp_info['frame_num'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    annotations = []
    for anno in json_data[video_name]['annotations']:
        start_frame = anno['segment'][0] * target_fps
        end_frame = anno['segment'][1] * target_fps
        label = anno['label']
        label_id = class_to_id[label]
        annotations.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'label': label,
            'label_id': label_id
        })
    tmp_info['annotations'] = annotations
    tmp_info['fps'] = target_fps
    new_video_info[video_name] = tmp_info

with open(new_video_info_path, 'w') as f:
    json.dump(new_video_info, f)

