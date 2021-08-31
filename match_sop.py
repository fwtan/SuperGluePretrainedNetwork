from pathlib import Path
import argparse, random, cv2, os
import numpy as np
import matplotlib.cm as cm
import torch
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from models.matching import Matching
from models.utils import frame2tensor
from utils import compute_metrics, read_file
from utils import pickle_load, pickle_save
from utils import json_load, json_save


torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching on SOP with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--nn_file', type=str)
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor', help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2, help='SuperGlue match threshold')
    parser.add_argument('--cache', action='store_true', help='Skip the pair if output .npz files are already found')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    opt = parser.parse_args()

    data_dir   = osp.join(opt.data_dir, 'Stanford_Online_Products')
    test_lines = read_file(osp.join(data_dir, 'test.txt'))
    global_nns = pickle_load(nn_file)
    global_nns = torch.from_numpy(global_nns).long()

    ############################################################################
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    #############################################################################

    #############################################################################
    # Create the output directories if they do not exist already.
    output_dir = osp.join('outputs', 'eval_sop')
    os.makedirs(output_dir, exist_ok=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))


    labels = np.array([x.split(',')[1] for x in test_lines]).astype(np.int32)
    num_samples = len(test_lines)
    top_k = 100
    ks = [1,10,100]
    ########################################################################################
    scores = np.zeros((num_samples, top_k), dtype=np.int32)
    for i in tqdm(range(num_samples)):
        src_pth = osp.join(data_dir, test_lines[i].split(',')[0])
        src_img = cv2.imread(src_pth, cv2.IMREAD_GRAYSCALE)
        src_img = cv2.resize(src_img, (320, 320))
        src_th  = frame2tensor(src_img.astype('float32'), device)
        # cv2.imwrite('%05d.jpg'%i, src_img)
        # if i > 1:
        #     break
        for j in range(top_k):
            matches_path = osp.join(output_dir, '%03d_%03d.npz'%(i, j))
            if opt.cache and osp.exists(matches_path):
                out_matches = pickle_load(matches_path)
                kpts0, kpts1, matches, conf = \
                    out_matches['keypoints0'], out_matches['keypoints1'], out_matches['matches'], out_matches['match_confidence']
            else:
                tgt_pth = osp.join(data_dir, test_lines[global_nns[i, j]].split(',')[0])
                tgt_img = cv2.imread(tgt_pth, cv2.IMREAD_GRAYSCALE)
                tgt_img = cv2.resize(tgt_img, (320, 320))
                tgt_th  = frame2tensor(tgt_img.astype('float32'), device)
                pred = matching({'image0': src_th, 'image1': tgt_th})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                            'matches': matches, 'match_confidence': conf}
                pickle_save(matches_path, out_matches)
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            scores[i, j] = len(mkpts0)
    
    scores = torch.from_numpy(scores)
    _, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(global_nns, -1, indices)
    
    max_k = max(ks)
    recalls = {}
    for k in ks:
        indices = closest_indices[:, :k]
        recalls[k] = (labels[:, None] == labels[indices]).any(1).mean()
    out = {k: round(v * 100, 2) for k, v in recalls.items()}
    print(out)
    pickle_save(osp.join(output_dir, 'superglue_sop.pkl'), out)