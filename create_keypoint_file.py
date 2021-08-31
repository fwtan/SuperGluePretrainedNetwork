import argparse, torch, cv2, os
import numpy as np
import os.path as osp
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from models.superpoint import SuperPoint
from models.utils import frame2tensor
from utils import pickle_load, pickle_save


torch.set_grad_enabled(False)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracting keypoints using SuperPoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_keypoints', type=int, default=500, help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--resize', type=int, nargs='+', default=[320, 320],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument('--cache', action='store_true', help='Skip the pair if output .npz files are already found')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    model = SuperPoint({'nms_radius': opt.nms_radius, 'keypoint_threshold': opt.keypoint_threshold, 'max_keypoints': opt.max_keypoints})
    model = model.eval().to(device)

    data_dir = 'data/Stanford_Online_Products'
    train_lines = read_file(osp.join(data_dir, 'train.txt'))
    test_lines  = read_file(osp.join(data_dir, 'test.txt'))
    lines = train_lines + test_lines
    output_dir = 'outputs/super_sop'
    os.makedirs(output_dir, exist_ok=True)
    print('Will write the detected points to directory \"{}\"'.format(output_dir))

    num_samples = len(lines)
    min_samples, max_samples = 1000, -1
    for i in tqdm(range(num_samples)):
        input_path = osp.join(data_dir, lines[i].split(',')[0])
        input_name = osp.splitext(osp.basename(lines[i]))[0]
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        input_image = cv2.resize(input_image, tuple(opt.resize)).astype('float32')
        input_image = frame2tensor(input_image, device)
        pred = model({'image': input_image})
        keypoints = pred['keypoints'][0]
        scores    = pred['scores'][0]
        keypoints[:, 0] /= float(opt.resize[0])
        keypoints[:, 1] /= float(opt.resize[1])
        min_samples = min(len(keypoints), min_samples)
        max_samples = max(len(keypoints), max_samples)
        out = {'keypoints': keypoints.cpu().data.numpy(), 'scores': scores.cpu().data.numpy()}
        out_path = osp.join(output_dir, '%s.pkl'%input_name)
        pickle_save(out_path, out)
    print('min, max', min_samples, max_samples)
    cache_files = sorted(glob(output_dir+'/*.pkl'))
    out = {}
    for path in tqdm(cache_files):
        name = osp.splitext(osp.basename(path))[0]
        out[name] = pickle_load(path)
    pickle_save(osp.join(output_dir, 'all_super_sop.pkl'), out)    