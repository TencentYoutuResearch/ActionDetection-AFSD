# AFSD for ActivityNet v1.3

## Data Pre-Process
Note that it needs at least 1TB disk space to save and pre-process ActivityNet dataset.
### RGB Data
1. Download original ActivityNet v1.3 videos and put them in `datasets/activitynet/v1-3/train_val`
2. Run the script to generate sampled videos: `python3 AFSD/anet_data/transform_videos.py THREAD_NUM`
3. Run the script to generate RGB npy input data: `python3 AFSD/anet_data/video2npy.py THREAD_NUM`
### Flow Data
1. Generate video list: `python3 AFSD/anet_data/gen_video_list.py`
2. Use [denseflow](https://github.com/open-mmlab/denseflow) to generate flow frames
3. Run the script to generate flow npy input data: `python3 AFSD/anet_data/flow2npy.py THREAD_NUM`

## Inference
TODO

## Training
TODO