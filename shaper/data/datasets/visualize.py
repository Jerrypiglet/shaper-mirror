import argparse
import math
import gc
from datetime import datetime
import h5py
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from shaper.utils.io import mkdir
from shaper.data.datasets.geometry_utils import *
import random
import json
from progressbar import ProgressBar
from subprocess import call
from scipy.optimize import linear_sum_assignment
from shutil import copyfile
from shaper.data import transforms as T






def gen_visu(visu_dir, dataset, pred_ins_label, conf_label, ious=None, visu_num=1000 ) :
    '''
    pts n_shape x num_point x 3 float32
    gt_ins_labels n_shape x num_point int32
    '''

    pts_dir = os.path.join(visu_dir, 'pts')
    pts_flipped_dir = os.path.join(visu_dir, 'pts_flipped')
    info_dir = os.path.join(visu_dir, 'info')
    mesh_dir = os.path.join(visu_dir, 'mesh')
    child_dir = os.path.join(visu_dir, 'child')

    mkdir(pts_dir)
    mkdir(pts_flipped_dir)
    mkdir(info_dir)
    mkdir(mesh_dir)
    mkdir(child_dir)

    bar = ProgressBar()
    n_shape = pred_ins_label.shape[0]
    n_shape = min(visu_num, n_shape)


    for i in bar(range(0,n_shape)):
        cur_fn_prefix = 'shape-%03d' % i
        data_dict=dataset[i]
        pts = data_dict['points']
        pts = np.transpose(pts,[1,0])
        pts_flipped = pts.copy()
        pts_flipped[:,0] *= -1
        pts_flipped[:,2] *= -1
        gt_ins_label = data_dict['ins_seg_label']
        record = data_dict['record']

        out_fn = os.path.join(pts_dir, cur_fn_prefix+'.png')
        merged_gt_ins_label = np.zeros((gt_ins_label.shape[1]))
        for j in range(gt_ins_label.shape[0]):
            merged_gt_ins_label[gt_ins_label[j,:].astype(np.bool_)] = j+1
        render_pts_with_label(out_fn, pts, merged_gt_ins_label)

        out_fn = os.path.join(pts_flipped_dir, cur_fn_prefix+'.png')
        render_pts_with_label(out_fn, pts_flipped, merged_gt_ins_label)

        out_fn = os.path.join(mesh_dir, cur_fn_prefix+'.png')
        copyfile('/media/ronald/bef0e123-8cc1-48ea-9dec-06f1293b847c/ronald/PartNet/data/shapenetpp_final_system/storage/downloads/%s/parts_render/0.png'%record['anno_id'],out_fn)

        out_fn = os.path.join(info_dir, cur_fn_prefix+'.txt')
        with open(out_fn,'w') as fout:
                fout.write('Anno_id: %s' % record['anno_id'] )



        cur_child_dir = os.path.join(child_dir, cur_fn_prefix)
        mkdir(cur_child_dir)
        child_part_dir = os.path.join(cur_child_dir, 'part')
        mkdir(child_part_dir)
        child_part_flipped_dir = os.path.join(cur_child_dir, 'part_flipped')
        mkdir(child_part_flipped_dir)
        child_info_dir = os.path.join(cur_child_dir, 'info')
        mkdir(child_info_dir)

        for j in range(pred_ins_label.shape[1]):
            if np.sum(pred_ins_label[i,j] > 0.1) == 0:
                continue
            if conf_label[i,j]<0.01:
                continue
            cur_part_prefix = 'part-%03d' % j
            out_fn = os.path.join(child_part_dir, cur_part_prefix+'.png')
            render_pts_with_feature(out_fn, pts, pred_ins_label[i,j])

            out_fn = os.path.join(child_part_flipped_dir, cur_part_prefix+'.png')
            render_pts_with_feature(out_fn, pts_flipped, pred_ins_label[i,j])


            out_fn = os.path.join(child_info_dir, cur_part_prefix+'.txt')
            with open(out_fn,'w') as fout:
                fout.write('conf: %f\n' % conf_label[i,j] )
                if ious is not None:
                    fout.write('iou: %f' % ious[i,j] )




def gen_foveal_visu(visu_dir, dataset, viewed_masks, proposal_logits, finish_logits, zoomed_points_all, pred_ins_label, conf_label, all_ret=None, all_conf=None, all_ious=None, visu_num=1000 ) :
    '''
    '''

    pts_dir = os.path.join(visu_dir, 'pts')
    pts_flipped_dir = os.path.join(visu_dir, 'pts_flipped')
    info_dir = os.path.join(visu_dir, 'info')
    mesh_dir = os.path.join(visu_dir, 'mesh')
    child_dir = os.path.join(visu_dir, 'child')

    mkdir(pts_dir)
    mkdir(pts_flipped_dir)
    mkdir(info_dir)
    mkdir(mesh_dir)
    mkdir(child_dir)

    bar = ProgressBar()
    num_zoom_iteration = len(pred_ins_label)
    n_shape = pred_ins_label[0].shape[0]
    n_shape = min(visu_num, n_shape)


    for i in bar(range(0,n_shape)):
        cur_fn_prefix = 'shape-%03d' % i
        data_dict=dataset[i]
        pts = data_dict['points']
        pts = np.transpose(pts, [1,0])


        pts_flipped = pts.copy()
        pts_flipped[:,0] *= -1
        pts_flipped[:,2] *= -1
        gt_ins_label = data_dict['ins_seg_label']
        record = data_dict['record']

        out_fn = os.path.join(pts_dir, cur_fn_prefix+'.png')
        merged_gt_ins_label = np.zeros((gt_ins_label.shape[1]))
        for j in range(gt_ins_label.shape[0]):
            merged_gt_ins_label[gt_ins_label[j,:].astype(np.bool_)] = j+1

        render_pts_with_label(out_fn, pts, merged_gt_ins_label)

        out_fn = os.path.join(pts_flipped_dir, cur_fn_prefix+'.png')
        render_pts_with_label(out_fn, pts_flipped, merged_gt_ins_label)

        out_fn = os.path.join(mesh_dir, cur_fn_prefix+'.png')
        copyfile('/media/ronald/bef0e123-8cc1-48ea-9dec-06f1293b847c/ronald/PartNet/data/shapenetpp_final_system/storage/downloads/%s/parts_render/0.png'%record['anno_id'],out_fn)

        out_fn = os.path.join(info_dir, cur_fn_prefix+'.txt')
        with open(out_fn,'w') as fout:
                fout.write('Anno_id: %s' % record['anno_id'] )
                fout.write('Model_id: %s' % record['model_id'] )


        cur_child_dir = os.path.join(child_dir, cur_fn_prefix)
        mkdir(cur_child_dir)
        child_part_dir = os.path.join(cur_child_dir, 'part')
        mkdir(child_part_dir)
        child_part_flipped_dir = os.path.join(cur_child_dir, 'part_flipped')
        mkdir(child_part_flipped_dir)
        child_info_dir = os.path.join(cur_child_dir, 'info')
        mkdir(child_info_dir)
        child_proposal_dir = os.path.join(cur_child_dir, 'proposal')
        mkdir(child_proposal_dir)
        child_proposal_flipped_dir = os.path.join(cur_child_dir, 'proposal_flipped')
        mkdir(child_proposal_flipped_dir)

        if all_ret is not None:
            big_mask = np.zeros((all_ret.shape[2],))
            for j in range(all_ret.shape[1]):
                if all_conf[i,j]==0:
                    continue
                big_mask[all_ret[i,j]>0]  = 1+j
                out_fn = os.path.join(child_part_dir, 'all_ret_%d.png'%j)
                render_pts_with_label(out_fn, pts, all_ret[i,j])
                out_fn = os.path.join(child_part_flipped_dir, 'all_ret_%d.png'%j)
                render_pts_with_label(out_fn, pts_flipped, all_ret[i,j])
                out_fn = os.path.join(child_info_dir, 'all_ret_%d.txt'%j)
                with open(out_fn,'w') as fout:
                    fout.write('conf: %f\n' % all_conf[i,j] )
                    fout.write('iou: %f\n' % all_ious[i,j] )


            out_fn = os.path.join(child_part_dir, 'all_merged.png')
            render_pts_with_label(out_fn, pts, big_mask)
            out_fn = os.path.join(child_part_flipped_dir, 'all_merged.png')
            render_pts_with_label(out_fn, pts_flipped, big_mask)



        for zoom_iteration in range(num_zoom_iteration):

            zoomed_points = np.transpose(zoomed_points_all[zoom_iteration], [0,2,1])
            zoomed_points_flipped = zoomed_points.copy()
            zoomed_points_flipped[:,:,0]*=-1
            zoomed_points_flipped[:,:,2]*=-1

            out_fn = os.path.join(child_part_dir, 'iteration%02d_stage1_proposal.png'%zoom_iteration)
            render_pts_with_feature(out_fn, pts, proposal_logits[zoom_iteration][i], normalize=True, fancy_kp=True)
            out_fn = os.path.join(child_part_flipped_dir, 'iteration%02d_stage1_proposal.png'%zoom_iteration)
            render_pts_with_feature(out_fn, pts_flipped, proposal_logits[zoom_iteration][i], normalize=True, fancy_kp=True)
            out_fn = os.path.join(child_info_dir, 'iteration%02d_stage1_proposal.txt'%zoom_iteration)
            with open(out_fn,'w') as fout:
                fout.write('finish: %f' % finish_logits[zoom_iteration][i] )

            if finish_logits[zoom_iteration][i] == 0:
                break

            out_fn = os.path.join(child_part_dir, 'iteration%02d_stage1_zoom.png'%zoom_iteration)
            render_pts_with_label(out_fn, pts, viewed_masks[zoom_iteration][i])
            out_fn = os.path.join(child_part_flipped_dir, 'iteration%02d_stage1_zoom.png'%zoom_iteration)
            render_pts_with_label(out_fn, pts_flipped, viewed_masks[zoom_iteration][i])

            for j in range(pred_ins_label[zoom_iteration].shape[1]):
                if conf_label[zoom_iteration][i,j]<0.01:
                    continue
                cur_part_prefix = 'iteration%02d_stage2_part-%03d' % (zoom_iteration, j)
                out_fn = os.path.join(child_part_dir, cur_part_prefix+'.png')
                render_pts_with_feature(out_fn, zoomed_points[i], pred_ins_label[zoom_iteration][i,j])

                out_fn = os.path.join(child_part_flipped_dir, cur_part_prefix+'.png')
                render_pts_with_feature(out_fn, zoomed_points_flipped[i], pred_ins_label[zoom_iteration][i,j])


                out_fn = os.path.join(child_info_dir, cur_part_prefix+'.txt')
                with open(out_fn,'w') as fout:
                    fout.write('conf: %f' % conf_label[zoom_iteration][i,j] )


