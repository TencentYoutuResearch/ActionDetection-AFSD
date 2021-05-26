# AFSD for ActivityNet v1.3

## Data Pre-Process
Note that it needs at least 1TB disk space to save and pre-process ActivityNet dataset.
### RGB Data
1. Download original ActivityNet v1.3 videos and put them in `datasets/activitynet/v1-3/train_val`
2. Run the script to generate sampled videos: `python3 AFSD/anet_data/transform_videos.py THREAD_NUM`
3. Run the script to generate RGB npy input data: `python3 AFSD/anet_data/video2npy.py THREAD_NUM`

In addition, the sampled videos (32.4GB) is provided: [\[Weiyun\]](https://share.weiyun.com/PXXtHcbp), and only run the step 3 to generate RGB npy data.

### Flow Data
1. Generate video list: `python3 AFSD/anet_data/gen_video_list.py`
2. Use [denseflow](https://github.com/open-mmlab/denseflow) to generate flow frames: 
`denseflow anet_anotations/anet_train_val.txt -b=20 -a=tvl1 -s=1 -o=datasets/activitynet/flow/frame_train_val_112`
3. Run the script to generate flow npy input data: `python3 AFSD/anet_data/flow2npy.py THREAD_NUM`

In addition, the flow frames (17.6GB) is provided: [\[Weiyun\]](https://share.weiyun.com/v3nI6EDv), and only run the step 3 to generate flow npy data.

## Inference
1. We provide the pretrained models contain final RGB and flow models for ActivityNet dataset:
[\[Google Drive\]](https://drive.google.com/drive/folders/1IG51-hMHVsmYpRb_53C85ISkpiAHfeVg?usp=sharing),
[\[Weiyun\]](https://share.weiyun.com/ImV5WYil)

2. Download CUHK validation action class results: [\[Google Drive\]](https://drive.google.com/drive/folders/1It9pGH-iM0gXMRVv_UxVo08vT15yeGFW?usp=sharing),
[\[Weiyun\]](https://share.weiyun.com/mkZl7rWK)

```shell script
# run RGB model 
python3 AFSD/anet/test.py configs/anet.yaml --output_json=anet_rgb.json --nms_sigma=0.85 --ngpu=GPU_NUM 

# run Flow model 
python3 AFSD/anet/test.py configs/anet_flow.yaml --output_json=anet_flow.json --nms_sigma=0.85 --ngpu=GPU_NUM 

# run RGB + Flow model
python3 AFSD/anet/test_fusion.py configs/anet.yaml --output_json=anet_fusion.json --nms_sigma=0.85 --ngpu=GPU_NUM
```
## Evaluation
The output json results of pretrained model can be downloaded from: [\[Google Drive\]](https://drive.google.com/drive/folders/10VCWQi1uXNNpDKNaTVnn7vSD9YVAp8ut?usp=sharing),
[\[Weiyun\]](https://share.weiyun.com/R7RXuFFW)
```shell script
# evaluate ActivityNet validation fusion result as example
python3 AFSD/anet/eval.py output/anet_fusion.json

mAP at tIoU 0.5 is 0.5238085847822328
mAP at tIoU 0.55 is 0.49477717170654223
mAP at tIoU 0.6 is 0.4644256093014668
mAP at tIoU 0.65 is 0.4308121487730952
mAP at tIoU 0.7 is 0.3962430306625962
mAP at tIoU 0.75 is 0.35270563112651215
mAP at tIoU 0.8 is 0.3006916408143017
mAP at tIoU 0.85 is 0.2421417273323893
mAP at tIoU 0.8999999999999999 is 0.16896798596919388
mAP at tIoU 0.95 is 0.06468751685005883
Average mAP: 0.34392610473183893
```

## Training
```shell script
# train RGB model
python3 AFSD/anet/train.py configs/anet.yaml --lw=1 --cw=1 --piou=0.6

# train Flow model
python3 AFSD/anet/train.py configs/anet_flow.yaml --lw=1 --cw=1 --piou=0.6
```