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
        description='Image pair matching on Revisited with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='oxford5k')
    parser.add_argument('--desc', type=str, default='r50_gldv2')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor', help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4, help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20, help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2, help='SuperGlue match threshold')
    parser.add_argument('--cache', action='store_true', help='Skip the pair if output .npz files are already found')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    opt = parser.parse_args()

    data_dir = osp.join(opt.data_dir, opt.dataset)
    query_lines = read_file(osp.join(data_dir, 'test_query.txt'))
    index_lines = read_file(osp.join(data_dir, 'test_gallery.txt'))
    global_nns  = pickle_load(osp.join(data_dir, 'nn_inds_%s.pkl'%opt.desc))
    gnd_data    = pickle_load(osp.join(data_dir, 'gnd_r%s.pkl'%opt.dataset))

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
    output_dir = osp.join('outputs', '%s_%s'%(opt.dataset, opt.desc))
    os.makedirs(output_dir, exist_ok=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))


    num_samples = len(query_lines)
    top_k = 100
    ########################################################################################
    ## Medium
    medium_nn_inds = deepcopy(global_nns)
    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    for i in range(num_samples):
        junk_ids = gnd_data['gnd'][i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids
    medium_nn_inds = torch.from_numpy(medium_nn_inds)

    scores = np.zeros((num_samples, top_k), dtype=np.int32)
    for i in tqdm(range(num_samples)):
        src_pth = osp.join(data_dir, query_lines[i].split(',')[0])
        src_img = cv2.imread(src_pth, cv2.IMREAD_GRAYSCALE)
        bbx = gnd_data['gnd'][i]['bbx']
        src_img = src_img[int(bbx[1]+0.5):int(bbx[3]+0.5)+1, int(bbx[0]+0.5):(int(bbx[2]+0.5)+1)]
        src_th  = frame2tensor(src_img.astype('float32'), device)
        # cv2.imwrite('%05d.jpg'%i, src_img)
        for j in range(top_k):
            matches_path = osp.join(output_dir, '%03d_%03d_medium.npz'%(i, j))
            if opt.cache and osp.exists(matches_path):
                out_matches = pickle_load(matches_path)
                kpts0, kpts1, matches, conf = \
                    out_matches['keypoints0'], out_matches['keypoints1'], out_matches['matches'], out_matches['match_confidence']
            else:
                tgt_pth = osp.join(data_dir, index_lines[medium_nn_inds[i, j]].split(',')[0])
                tgt_img = cv2.imread(tgt_pth, cv2.IMREAD_GRAYSCALE)
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
    closest_indices = torch.gather(medium_nn_inds, -1, indices)
    ranks = deepcopy(medium_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    medium = compute_metrics('revisited', ranks, gnd_data['gnd'], kappas=[1,5,10])
    pickle_save(osp.join(output_dir, 'medium_score.pkl'), medium)


    ########################################################################################
    ## Hard
    hard_nn_inds = deepcopy(global_nns)
    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    for i in range(num_samples):
        junk_ids = gnd_data['gnd'][i]['junk'] + gnd_data['gnd'][i]['easy']
        all_ids = hard_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        hard_nn_inds[i] = new_ids
    hard_nn_inds = torch.from_numpy(hard_nn_inds)


    scores = np.zeros((num_samples, top_k), dtype=np.int32)
    for i in tqdm(range(num_samples)):
        src_pth = osp.join(data_dir, query_lines[i].split(',')[0])
        src_img = cv2.imread(src_pth, cv2.IMREAD_GRAYSCALE)
        bbx = gnd_data['gnd'][i]['bbx']
        src_img = src_img[int(bbx[1]+0.5):int(bbx[3]+0.5)+1, int(bbx[0]+0.5):(int(bbx[2]+0.5)+1)]
        src_th  = frame2tensor(src_img.astype('float32'), device)
        # cv2.imwrite('%05d.jpg'%i, src_img)
        for j in range(top_k):
            matches_path = osp.join(output_dir, '%03d_%03d_hard.npz'%(i, j))
            if opt.cache and osp.exists(matches_path):
                out_matches = pickle_load(matches_path)
                kpts0, kpts1, matches, conf = \
                    out_matches['keypoints0'], out_matches['keypoints1'], out_matches['matches'], out_matches['match_confidence']
            else:
                tgt_pth = osp.join(data_dir, index_lines[hard_nn_inds[i, j]].split(',')[0])
                tgt_img = cv2.imread(tgt_pth, cv2.IMREAD_GRAYSCALE)
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
    closest_indices = torch.gather(hard_nn_inds, -1, indices)
    ranks = deepcopy(hard_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    hard = compute_metrics('revisited', ranks, gnd_data['gnd'], kappas=[1,5,10])
    pickle_save(osp.join(output_dir, 'hard_score.pkl'), hard)
