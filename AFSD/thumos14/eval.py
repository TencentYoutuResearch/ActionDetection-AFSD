import argparse
from AFSD.evaluation.eval_detection import ANETdetection

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str, default='./thumos_annotations/thumos_gt.json', nargs='?')
args = parser.parse_args()

tious = [0.3, 0.4, 0.5, 0.6, 0.7]
anet_detection = ANETdetection(
    ground_truth_filename=args.gt_json,
    prediction_filename=args.output_json,
    subset='test', tiou_thresholds=tious)
mAPs, average_mAP, ap = anet_detection.evaluate()
for (tiou, mAP) in zip(tious, mAPs):
    print("mAP at tIoU {} is {}".format(tiou, mAP))
