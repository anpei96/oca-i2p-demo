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

# isort: split
from .fusion_module    import CrossModalFusionModule
from .image_backbone   import FeaturePyramid, ImageBackbone
from .point_backbone   import PointBackbone
from .base_model_multi import baseI2P
from .match_utils import pairwiseL2Dist, RegularisedTransport
from .transformer import MultiHeadAttention
from .match_utils import pairwiseL2Dist, RegularisedTransport
from .utils_ode   import ode_att_layer

class baseI2P_multiview_norgb(baseI2P):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        self.img_h_c = 24//2
        self.img_w_c = 32//2
        self.img_num_levels_c = 3
        self.overlap_threshold = 0.3
        self.pcd_min_node_size = 5

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim,
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation)

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim,
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma)
        
        self.transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding)

        self.img_pyramid = FeaturePyramid(cfg.model.transformer.output_dim)

        '''
            note-0109
                adding the module of ode-att
        '''
        self.is_enable_ode_att = False
        # self.is_enable_ode_att = True
        
    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}

        init_gpu_memory = torch.cuda.memory_allocated()
        
        # 1. Unpack data from data dict
        img_feats, img_feats_next, img_feats_prev, output_dict \
            = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = output_dict["pcd_points"].detach() # no rgb features

        # 2. Backbone
        img_feats_list = self.img_backbone(img_feats)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)
        pcd_feats_c_raw = pcd_feats_c * 1.0

        # camera view #1
        img_feats_list_next = self.img_backbone(img_feats_next)
        img_feats_x_next    = img_feats_list_next[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f_next    = img_feats_list_next[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)

        # camera view #2
        img_feats_list_prev = self.img_backbone(img_feats_prev)
        img_feats_x_prev    = img_feats_list_prev[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f_prev    = img_feats_list_prev[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")

        # 3. Transformer
        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)

        # ===> camera view 1
        img_feats_c_next = F.interpolate(
            img_feats_x_next, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c_next = \
            img_feats_c_next.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)
        # ===> camera view 2
        img_feats_c_prev = F.interpolate(
            img_feats_x_prev, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c_prev = img_feats_c_prev.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)

        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c_raw.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0))
        
        # ===> camera view 1
        img_feats_c_next, pcd_feats_c_next = self.transformer(
            img_feats_c_next.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c_raw.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0))

        # ===> camera view 2
        img_feats_c_prev, pcd_feats_c_prev = self.transformer(
            img_feats_c_prev.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c_raw.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0))

        # 3.3 Post-transformer image feature pyramid
        img_feats_c = img_feats_c.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        all_img_feats_c = self.img_pyramid(img_feats_c)
        all_img_feats_c = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c]
        img_feats_c = torch.cat(all_img_feats_c, dim=0)

        # ===> camera view 1
        img_feats_c_next = img_feats_c_next.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        all_img_feats_c_next = self.img_pyramid(img_feats_c_next)
        all_img_feats_c_next = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c_next]
        img_feats_c_next = torch.cat(all_img_feats_c_next, dim=0)

        # ===> camera view 2
        img_feats_c_prev = img_feats_c_prev.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        all_img_feats_c_prev = self.img_pyramid(img_feats_c_prev)
        all_img_feats_c_prev = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c_prev]
        img_feats_c_prev = torch.cat(all_img_feats_c_prev, dim=0)

        # 4. Coarse-level matching
        pcd_feats_c = pcd_feats_c.squeeze(0)
        '''
            note-0109
                adding the module of ode-att
                camera view 0
        '''
        if self.is_enable_ode_att == True:
            img_feats_c, pcd_feats_c = \
                ode_att_layer(img_feats_c, pcd_feats_c)

        img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        # ===> camera view 1
        pcd_feats_c_next = pcd_feats_c_next.squeeze(0)
        '''
            note-0109
                adding the module of ode-att
                camera view 1
        '''
        if self.is_enable_ode_att == True:
            img_feats_c_next, pcd_feats_c_next = \
                ode_att_layer(img_feats_c_next, pcd_feats_c_next)

        img_feats_c_next = F.normalize(img_feats_c_next, p=2, dim=1)
        pcd_feats_c_next = F.normalize(pcd_feats_c_next, p=2, dim=1)

        # ===> camera view 2
        pcd_feats_c_prev = pcd_feats_c_prev.squeeze(0)
        '''
            note-0109
                adding the module of ode-att
                camera view 2
        '''
        if self.is_enable_ode_att == True:
            img_feats_c_prev, pcd_feats_c_prev = \
                ode_att_layer(img_feats_c_prev, pcd_feats_c_prev)

        img_feats_c_prev = F.normalize(img_feats_c_prev, p=2, dim=1)
        pcd_feats_c_prev = F.normalize(pcd_feats_c_prev, p=2, dim=1)

        output_dict["img_feats_c"]   = img_feats_c
        output_dict["pcd_feats_c"]   = pcd_feats_c
        output_dict["img_feats_c_next"] = img_feats_c_next
        output_dict["pcd_feats_c_next"] = pcd_feats_c_next
        output_dict["img_feats_c_prev"] = img_feats_c_prev
        output_dict["pcd_feats_c_prev"] = pcd_feats_c_prev

        '''
            a crucial step to generate 2d-3d corresponding labels
                due to data augmentation in dataloader

            todo ======> please add multiview 2d-3d patches gt labels
        '''
        an_time_start = time.time()
        output_dict = self.genenrate_label(output_dict)
        output_dict = self.genenrate_label_prev(output_dict)
        output_dict = self.genenrate_label_next(output_dict)
        an_time_end = time.time()
       
        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        # ===> camera view 1
        img_feats_f_next = \
            img_feats_f_next.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()
        # ===> camera view 2
        img_feats_f_prev = \
            img_feats_f_prev.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)
        
        # ===> camera view 1
        img_feats_f_next = F.normalize(img_feats_f_next, p=2, dim=1)
        # ===> camera view 2
        img_feats_f_prev = F.normalize(img_feats_f_prev, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        # ===> camera view 1
        output_dict["img_feats_f_next"] = img_feats_f_next
        output_dict["pcd_feats_f_next"] = pcd_feats_f
        # ===> camera view 2
        output_dict["img_feats_f_prev"] = img_feats_f_prev
        output_dict["pcd_feats_f_prev"] = pcd_feats_f

        # final_gpu_memory = torch.cuda.memory_allocated()
        # print("==> memory usage in main branch: ", 
        #     (final_gpu_memory-init_gpu_memory)/1e6, " MB")

        # 6. Select topk nearest node correspondences
        if not self.training:
            output_dict = self.post_process_generate_corres(
                img_feats_c.detach(), pcd_feats_c.detach(), 
                img_feats_f.detach(), pcd_feats_f.detach(), output_dict)
            '''
                add multiview 2d-3d patches gt labels
            '''
            output_dict = self.post_process_generate_corres_next(
                img_feats_c_next.detach(), pcd_feats_c_next.detach(), 
                img_feats_f_next.detach(), pcd_feats_f.detach(), output_dict)
            output_dict = self.post_process_generate_corres_prev(
                img_feats_c_prev.detach(), pcd_feats_c_prev.detach(), 
                img_feats_f_prev.detach(), pcd_feats_f.detach(), output_dict)

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        print("=======> cost time (model inference): ", duration-(an_time_end-an_time_start))
        return output_dict

def create_model(cfg):
    model = baseI2P_multiview_norgb(cfg)
    return model

