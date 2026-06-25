from this import d
import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, cast
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.layers import ConvBlock, build_act_layer
from vision3d.array_ops import axis_angle_to_rotation_matrix, get_transform_from_rotation_translation
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render)
from torchvision.transforms import Compose
from vision3d.ops import pairwise_distance

def prune_att_matrix(att, k=5):
    num_rows, num_cols = att.shape

    row_topk_indices = att.topk(k=k, largest=False, dim=1)[1]  # (N, K)
    row_indices = torch.arange(num_rows).cuda().view(num_rows, 1).expand(-1, k)  # (N, K)
    row_corr_mat = torch.zeros_like(att, dtype=torch.bool)  # (N, M)
    row_corr_mat[row_indices, row_topk_indices] = True

    col_topk_indices = att.topk(k=k, largest=False, dim=0)[1]  # (K, M)
    col_indices = torch.arange(num_cols).cuda().view(1, num_cols).expand(k, -1)  # (K, M)
    col_corr_mat = torch.zeros_like(att, dtype=torch.bool)  # (N, M)
    col_corr_mat[col_topk_indices, col_indices] = True
    corr_mat = torch.logical_or(row_corr_mat, col_corr_mat)

    att_topk = att #* corr_mat
    att_topk = att_topk/(torch.sum(att_topk, dim=1, keepdim=True)+1e-5)
    
    att_topk_t = att_topk.T
    att_topk_t = att_topk_t/(torch.sum(att_topk_t, dim=1, keepdim=True)+1e-5)
    return att_topk, att_topk_t

def prune_att_matrix_v2(att, temperature=0.5):
    att_sp   = torch.softmax(att/temperature,   dim=1)
    att_sp_t = torch.softmax(att.T/temperature, dim=1)
    return att_sp, att_sp_t

def prop_att_once(x0, y0, xx0, yy0, att0, att0_sp, att0_sp_t, tau):
    att1 = att0 + \
        tau*(torch.matmul(xx0, att0_sp) + torch.matmul(att0_sp, yy0))
    x1   = x0 + tau*torch.matmul(att0_sp,   y0)
    y1   = y0 + tau*torch.matmul(att0_sp_t, x0)
    xx1  = torch.matmul(x1, x1.T) # [n,n]
    yy1  = torch.matmul(y1, y1.T) # [m,m]
    # x1_no = F.normalize(x1, p=2, dim=1)
    # y1_no = F.normalize(y1, p=2, dim=1)
    # att1  = torch.matmul(x1_no, y1_no.T) # [n,m]
    att1_sp, att1_sp_t = prune_att_matrix(att1, k=5)
    return x1, y1, xx1, yy1, att1, att1_sp, att1_sp_t

def prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau):
    diff_att = torch.matmul(xx0, rho_att0) + torch.matmul(rho_att0, yy0)
    att1 = att0 + tau*diff_att

    x1   = x0 + tau*torch.matmul(att1,   y0)
    y1   = y0 + tau*torch.matmul(att1.T, x0)
    # xx1  = torch.matmul(x1, x1.T) # [n,n]
    # yy1  = torch.matmul(y1, y1.T) # [m,m]
    x1_no = F.normalize(x1, p=2, dim=1)
    y1_no = F.normalize(y1, p=2, dim=1)
    xx1   = torch.matmul(x1_no, x1_no.T) # [n,n]
    yy1   = torch.matmul(y1_no, y1_no.T) # [m,m]
    # att1  = torch.matmul(x1_no, y1_no.T) # [n,m]
    att1_sp, att1_sp_t = prune_att_matrix_v2(att1, temperature=0.5)
    return x1, y1, xx1, yy1, att1, att1_sp, att1_sp_t

def ode_att_layer(img_feats_c, pcd_feats_c):
    # step 1. obtain the initial attention matrix
    x0    = img_feats_c                  # [n,c]
    y0    = pcd_feats_c                  # [m,c]
    # xx0   = torch.matmul(x0, x0.T)       # [n,n]
    # yy0   = torch.matmul(y0, y0.T)       # [m,m]
    x0_no = F.normalize(x0, p=2, dim=1)
    y0_no = F.normalize(y0, p=2, dim=1)
    xx0   = torch.matmul(x0_no, x0_no.T) # [n,n]
    yy0   = torch.matmul(y0_no, y0_no.T) # [m,m]
    att0  = torch.matmul(x0_no, y0_no.T) # [n,m]
    rho_att0, rho_att0_t = prune_att_matrix_v2(att0, temperature=0.5)
    # rho_att0, rho_att0_t = prune_att_matrix_v2(att0, temperature=2.0) # used for scannet

    # step 2. propagate the attention matrix
    tau=0.10 # 0.1
    x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
        prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
        prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
        prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    
    '''
        note-0120
            adding the visulization module for attention matrix
    '''
    # is_need_vis_att = True
    # if is_need_vis_att:
    #     # x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
    #     #     prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    #     vis_mat_0 = xx0
    #     x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
    #         prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    #     vis_mat_1 = att0
    #     x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t = \
    #         prop_att_once_v2(x0, y0, xx0, yy0, att0, rho_att0, rho_att0_t, tau)
    #     vis_mat_2 = att0

        # import numpy as np
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # vis_mat_0 = vis_mat_0.detach().cpu().numpy()
        # vis_mat_1 = vis_mat_1.detach().cpu().numpy()
        # vis_mat_2 = vis_mat_2.detach().cpu().numpy()
        
        # # Set up plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(vis_mat_0)
        # fig.colorbar(cax)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(vis_mat_1)
        # fig.colorbar(cax)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(vis_mat_2)
        # fig.colorbar(cax)
        # plt.show()

        # assert 1==-1
    
    # step 3. output the results
    alpha = 0.10
    img_feats_c = img_feats_c + alpha*x0
    pcd_feats_c = pcd_feats_c + alpha*y0
    return img_feats_c, pcd_feats_c