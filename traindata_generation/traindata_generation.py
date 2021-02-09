from multiprocessing import Pool
import os
import sys
import json
import glob
import time
import numpy as np
import random
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.join(os.path.dirname(__file__), "./proto/"))
import detection_results_pb2
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

_MAX_TRACKLET_NUM = 2048
_SPLIT_FRAME_GAP = 4
_SPLIT_IoU_GAP = 0.2
_SPLIT_reid_GAP = 0.1
_SPLIT_random_min_track_length = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_file_path', type=str, help='video detection pb file',
        required=True)
    parser.add_argument('--gt_file_path', type=str, help= 'gt json file',
        required=True)
    parser.add_argument('--num_proc', type=int, default=1, help='number of processors')
    parser.add_argument('--output_path', type=str, required=True,
        help='the path to save tracking results')
    return parser.parse_args()


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to length 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def Randomly_split_list(num, objList):
    if num == 1:
        return [objList]
    totalNum = len(objList)
    numList = sorted(list(set([np.random.randint(1, totalNum) for ii in range(num-1)])))
    split_objList = []
    for i, num in enumerate(numList):
        if i == 0:
            split_objList.append(objList[:num])
        else:
            split_objList.append(objList[numList[i-1]:num])
    if len(objList[num:]) > 0:
        split_objList.append(objList[num:])
    return split_objList


def iou(bbox, candidates, eps=0.00001):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:4]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:4].prod(axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection+eps)


def assign_matched_labels(track_nodes, label_nodes, overlap_threshold = 0.5):
    ret = {}
    ret_pid = {}
    for frame in sorted(track_nodes.keys()):
        if not track_nodes[frame]:
            continue
        if frame not in label_nodes or not label_nodes[frame]:
            continue
        tboxes = np.array([ d[0] for d in track_nodes[frame] ])
        lboxes = np.array([ d[0] for d in label_nodes[frame] ])
        ov = np.zeros((tboxes.shape[0], lboxes.shape[0]), dtype=np.float32)
        for i, box in enumerate(tboxes):
            ov[i, :] = iou(box, lboxes)
        #indices = linear_assignment(1.0 - ov)
        row_ind, col_ind = linear_sum_assignment(1.0 - ov)
        #for ij in indices:
        for i, j in zip(row_ind, col_ind):
            #i,j = ij
            pid = label_nodes[frame][j][1]
            if ov[i, j] > overlap_threshold and pid > -1:
                ret.setdefault(frame, []).append([track_nodes[frame][i][0], track_nodes[frame][i][1], pid])
                ret_pid.setdefault(pid, {})
                ret_pid[pid][frame] = track_nodes[frame][i]
    return ret, ret_pid


def assign_gt_to_detections(detection_file, gt_file):
    detections_pb = detection_results_pb2.Detections()
    with open(detection_file, 'rb') as f:
        detections_pb.ParseFromString(f.read())
    detections_from_file = {}
    for detection in detections_pb.tracked_detections:
        frame = detection.frame_index
        box = [detection.box_x, detection.box_y, detection.box_width, detection.box_height]
        feat = [ d for d in detection.features.features[0].feats ]
        assert len(feat) == 2048
        detections_from_file.setdefault(frame, []).append([box, feat])
    gt_from_file = {}
    with open(gt_file, 'r+') as f1:
        for line in f1:
            infos = line.rstrip()
            infos = infos.split(',')
            frame = int(infos[0])
            pid = int(infos[1])
            bbx = [float(infos[2]), float(infos[3]), float(infos[4]), float(infos[5])]
            gt_from_file.setdefault(frame, []).append([bbx, pid])
    ret, ret_pid = assign_matched_labels(detections_from_file, gt_from_file)
    return ret, ret_pid


def generate_tracklet(tracking_res, video_name):
    # We need to split the long trajectory of each pid to short tracklets
    # Firstly, we split it according to the frame gap, reid similarity and IoU.
    splited_tracking_res, tracklet_num = {}, 0
    for pid in sorted(tracking_res.keys()):
        tracking_res_pid = tracking_res[pid]
        frames_pid = sorted(list(tracking_res_pid.keys()))
        assert len(frames_pid) > 0, "tracking data has empty pid!"
        splited_frames_pid = []
        for i, frame in enumerate(frames_pid):
            if i == 0:
                splited_frames_pid.append([frame])
            else:
                box_prev_frame, box_cur_frame = tracking_res[pid][frames_pid[i-1]][0], tracking_res[pid][frames_pid[i]][0]
                iou_value = iou(np.array(box_prev_frame), np.array([box_cur_frame]))[0]
                reid_prev_frame, reid_cur_frame = tracking_res[pid][frames_pid[i-1]][1], tracking_res[pid][frames_pid[i]][1]
                reid_distance = cosine_distance(np.array([reid_prev_frame]), np.array([reid_cur_frame]))[0][0]
                if (frame - splited_frames_pid[-1][-1]) > _SPLIT_FRAME_GAP or iou_value < _SPLIT_IoU_GAP or reid_distance > _SPLIT_reid_GAP:
                    splited_frames_pid.append([frame])
                else:
                    splited_frames_pid[-1].append(frame)
        tracklet_num += len(splited_frames_pid)
        for tt in splited_frames_pid:
            assert len(tt) > 0, "First round of split: cause empty tracklet"
        splited_tracking_res[pid] = splited_frames_pid
    mlog.info("{}: First round of split: pid {} -> tracklet {}".format(video_name, len(tracking_res), tracklet_num))
    # Secondly, we randomly split the tracklets
    splited_tracking_res_randomly, tracklet_num_second = {}, 0
    if tracklet_num < _MAX_TRACKLET_NUM:
        for pid in sorted(splited_tracking_res.keys()):
            frames_pid_list = splited_tracking_res[pid]
            splited_frames_pid_list = []
            for frames_list in frames_pid_list:
                if len(frames_list) <= _SPLIT_random_min_track_length:
                    splited_frames_pid_list.append(frames_list)
                else:
                    split_number = np.random.randint(0, len(frames_list)//_SPLIT_random_min_track_length)
                    split_objList = Randomly_split_list(split_number, sorted(frames_list))
                    for tt in split_objList:
                        assert len(tt) > 0, "Second round of split: cause empty tracklet"
                    splited_frames_pid_list += split_objList
            splited_tracking_res_randomly[pid] = splited_frames_pid_list
            tracklet_num_second += len(splited_frames_pid_list)
        splited_tracking_res = splited_tracking_res_randomly
    else:
        tracklet_num_second = tracklet_num
    mlog.info("{}: Second round of split: pid {} -> tracklet {}".format(video_name, len(tracking_res), tracklet_num_second))
    return splited_tracking_res


def generate_proposals(detection_res, tracking_res):
    pure_proposals, impure_proposals = [], []
    # generate pure proposals
    pure_proposals_per_pid = {}
    for pid in sorted(tracking_res.keys()):
        tracklets = tracking_res[pid]
        proposal_used = []
        if len(tracklets) > 1:
            proposal_candidates = [sorted(random.sample(tracklets, random.randint(2, len(tracklets)))) for _ in range(100)]
            for proposal_candidate in proposal_candidates:
                assert len(proposal_candidate) > 1
                if proposal_candidate not in proposal_used:
                    proposal_used.append(proposal_candidate)
                    tracklet_features_reid, tracklet_features_spatem, adjcent_matrix = [], [], np.eye(len(proposal_candidate))
                    for ii, tracklet in enumerate(proposal_candidate):
                        average_reids = np.mean(np.array([detection_res[pid][frame][1] for frame in tracklet]), axis=0).tolist()
                        tracklet_features_reid.append(average_reids)
                        if ii == 0:
                            spatem_features = [1, 0, 0, 0, 0]
                        else:
                            fps = 10
                            time_diff = float(proposal_candidate[ii][0] - proposal_candidate[ii-1][-1])/fps
                            box_new, box_old = detection_res[pid][proposal_candidate[ii][0]][0], detection_res[pid][proposal_candidate[ii-1][-1]][0]
                            u_diff = float(box_new[0] - box_old[0])/((box_new[0] +  box_old[0])/2.0)
                            v_diff = float(box_new[1] - box_old[1])/((box_new[1] +  box_old[1])/2.0)
                            w_diff = np.log(float(box_new[2]) / box_old[2])
                            h_diff = np.log(float(box_new[3]) / box_old[3])
                            spatem_features = [time_diff, u_diff, v_diff, w_diff, h_diff]
                        tracklet_features_spatem.append(spatem_features)
                    for ii in range(len(proposal_candidate)):
                        for jj in range(ii, len(proposal_candidate)):
                            fea1, fea2 = tracklet_features_reid[ii], tracklet_features_reid[jj]
                            if jj == ii:
                                dist = 1.0
                            else:
                                reid_simi = (1 + np.dot(fea1,fea2)/(np.linalg.norm(fea1)*(np.linalg.norm(fea1)))) / 2.0
                                temporal_dist = np.exp(-1*(float(abs(proposal_candidate[jj][0] - proposal_candidate[ii][-1])/100)))
                                box_new, box_old = detection_res[pid][proposal_candidate[jj][0]][0], detection_res[pid][proposal_candidate[ii][-1]][0]
                                spatial_dist = np.exp(-1*np.linalg.norm(np.array(box_new[:2]) - np.array(box_old[:2])) / 200.0)
                                dist = (temporal_dist + spatial_dist + reid_simi)/3.0
                            adjcent_matrix[ii][jj] = dist
                            adjcent_matrix[jj][ii] = dist
                    proposal_info = {"reid_features": tracklet_features_reid, "spatem_features": tracklet_features_spatem, "labels": 1, "adjcent_matrix": adjcent_matrix.tolist()}
                    pure_proposals.append(proposal_info)
        if len(proposal_used) > 1:
            pure_proposals_per_pid[pid] = proposal_used
    # generate impure proposals
    while len(impure_proposals) < len(pure_proposals):
        pids_all = list(sorted(pure_proposals_per_pid.keys()))
        selected_pids = random.sample(pids_all, 2)
        selected_proposal_in_each_pid = [random.sample(pure_proposals_per_pid[pid], 1)[0] for pid in selected_pids]
        frames_pid1 = [frame for tracklet in selected_proposal_in_each_pid[0] for frame in tracklet]
        frames_pid2 = [frame for tracklet in selected_proposal_in_each_pid[1] for frame in tracklet]
        if list(set(frames_pid1)&set(frames_pid2)):
            continue
        proposal_candidate, pids = [], []
        for ii, selected_proposal in enumerate(selected_proposal_in_each_pid):
            for tracklet in selected_proposal:
                proposal_candidate.append(tracklet)
                pids.append(selected_pids[ii])
        tracklet_features_reid, tracklet_features_spatem, adjcent_matrix = [], [], np.eye(len(proposal_candidate))
        proposal_candidate1 = [tracklet for tracklet, _ in sorted(zip(proposal_candidate, pids))]
        pids1 = [pid for _, pid in sorted(zip(proposal_candidate, pids))]
        proposal_candidate, pids = proposal_candidate1, pids1
        for ii, tracklet in enumerate(proposal_candidate):
            pid = pids[ii]
            average_reids = np.mean(np.array([detection_res[pid][frame][1] for frame in tracklet]), axis=0).tolist()
            tracklet_features_reid.append(average_reids)
            if ii == 0:
                spatem_features = [1, 0, 0, 0, 0]
            else:
                fps = 10
                time_diff = float(proposal_candidate[ii][0] - proposal_candidate[ii-1][-1])/fps
                box_new, box_old = detection_res[pid][proposal_candidate[ii][0]][0], detection_res[pids[ii-1]][proposal_candidate[ii-1][-1]][0]
                u_diff = float(box_new[0] - box_old[0])/((box_new[0] +  box_old[0])/2.0)
                v_diff = float(box_new[1] - box_old[1])/((box_new[1] +  box_old[1])/2.0)
                w_diff = np.log(float(box_new[2]) / box_old[2])
                h_diff = np.log(float(box_new[3]) / box_old[3])
                spatem_features = [time_diff, u_diff, v_diff, w_diff, h_diff]
            tracklet_features_spatem.append(spatem_features)
        for ii in range(len(proposal_candidate)):
            for jj in range(ii, len(proposal_candidate)):
                fea1, fea2 = tracklet_features_reid[ii], tracklet_features_reid[jj]
                if jj == ii:
                    dist = 1.0
                else:
                    reid_simi = (1 + np.dot(fea1,fea2)/(np.linalg.norm(fea1)*(np.linalg.norm(fea1)))) / 2.0
                    temporal_dist = np.exp(-1*(float(abs(proposal_candidate[jj][0] - proposal_candidate[ii][-1])/100)))
                    box_new, box_old = detection_res[pids[jj]][proposal_candidate[jj][0]][0], detection_res[pids[ii]][proposal_candidate[ii][-1]][0]
                    spatial_dist = np.exp(-1*np.linalg.norm(np.array(box_new[:2]) - np.array(box_old[:2])) / 200.0)
                    dist = (temporal_dist + spatial_dist + reid_simi)/3.0
                adjcent_matrix[ii][jj] = dist
                adjcent_matrix[jj][ii] = dist
        proposal_info = {"reid_features": tracklet_features_reid, "spatem_features": tracklet_features_spatem, "labels": 0, "adjcent_matrix": adjcent_matrix.tolist()}
        impure_proposals.append(proposal_info)
    return pure_proposals, impure_proposals


def generate_traindata_single_video(context):
    try:
        detection_file, args = context
        gt_file = glob.glob(os.path.join(args.gt_file_path, os.path.basename(detection_file).split(".")[0] + "/gt/gt.txt"))
        video_name = os.path.basename(detection_file).split(".")[0]
        if len(gt_file) == 1:
            _, detection_res_with_label = assign_gt_to_detections(detection_file, gt_file[0])
            pure_proposals_all, impure_proposals_all = [], []
            while len(pure_proposals_all) < 5000:
                tracking_res = generate_tracklet(detection_res_with_label, video_name)
                pure_proposals, impure_proposals = generate_proposals(detection_res_with_label, tracking_res)
                pure_proposals_all += pure_proposals
                impure_proposals_all += impure_proposals
            mlog.info("file {}: {} pure and {} impure proposals".format(os.path.basename(detection_file).split(".")[0], len(pure_proposals_all), len(impure_proposals_all)))
            output_path = os.path.join(args.output_path, os.path.basename(detection_file).split(".")[0])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            impure_output_path = os.path.join(output_path, "impure")
            pure_output_path = os.path.join(output_path, "pure")
            if not os.path.exists(impure_output_path):
                os.makedirs(impure_output_path)
            if not os.path.exists(pure_output_path):
                os.makedirs(pure_output_path)
            for ii, proposal in enumerate(pure_proposals_all):
                output_file_name = os.path.join(pure_output_path, "{}.json".format(ii+1))
                json.dump(proposal, open(output_file_name, "w"))
            for ii, proposal in enumerate(impure_proposals_all):
                output_file_name = os.path.join(impure_output_path, "{}.json".format(ii+1))
                json.dump(proposal, open(output_file_name, "w"))
    except Exception as e:
        mlog.info(e)
        sys.exit()



if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    detection_files = sorted(glob.glob(os.path.join(args.detection_file_path, "*.pb")))
    if args.num_proc == 1:
        for detection_file in detection_files:
            generate_traindata_single_video((detection_file, args))
    else:
        p = Pool(args.num_proc)
        p.map(generate_traindata_single_video, [(detection_file, args) for i, detection_file in enumerate(detection_files)])
        p.close()
        p.join()
    mlog.info('all finished')
