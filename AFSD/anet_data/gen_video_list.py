import glob
import numpy as np

video_list = sorted(glob.glob('datasets/activitynet/train_val_112/*.mp4'))

np.savetxt('anet_anotations/anet_train_val.txt', video_list, '%s', '\n')
