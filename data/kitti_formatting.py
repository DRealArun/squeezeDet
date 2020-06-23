import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='KITTI dataset formatting script')
parser.add_argument('--data_path', default=None, action='store', type=str, help='Root directory of the KITTI dataset')

args = parser.parse_args()

assert args.data_path != None, "Data path not provided !"
assert os.path.exists(args.data_path), "Invalid Data path provided !"

image_set_dir = os.path.join(args.data_path, 'ImageSets')
if not os.path.exists(image_set_dir):
	os.mkdir(image_set_dir, 0o777)

trainval_file = os.path.join(image_set_dir, 'trainval.txt')
train_file = os.path.join(image_set_dir, 'train.txt')
val_file = os.path.join(image_set_dir, 'val.txt')
training_data_dir = os.path.join(args.data_path, 'training', 'image_2', '*.png')

with open(trainval_file, 'w') as f:
    for i in glob.iglob(training_data_dir):
        f.write(i.split('\\')[-1].split('.png')[0]+"\n") 

idx = []
with open(trainval_file) as f:
  for line in f:
    idx.append(line.strip())
f.close()

idx = np.random.permutation(idx)

train_idx = sorted(idx[:len(idx)//2])
val_idx = sorted(idx[len(idx)//2:])

with open(train_file, 'w') as f:
  for i in train_idx:
    f.write('{}\n'.format(i))
f.close()

with open(val_file, 'w') as f:
  for i in val_idx:
    f.write('{}\n'.format(i))
f.close()

print('Trainining set is saved to ' + train_file)
print('Validation set is saved to ' + val_file)
