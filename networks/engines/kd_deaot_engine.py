import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.engines.aot_engine import AOTEngine, AOTInferEngine


class KD_DeAOTEngine(AOTEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 layer_loss_scaling_ratio=2.):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip)
        self.layer_loss_scaling_ratio = layer_loss_scaling_ratio


    # def _init_losses(self):
    #     cfg = self.cfg
    #
    #     from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
    #     from networks.layers.dist_kd import DIST
    #     bce_loss = CrossEntropyLoss(
    #         cfg.TRAIN_TOP_K_PERCENT_PIXELS,
    #         cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
    #     iou_loss = SoftJaccordLoss()
    #
    #     losses = [bce_loss, iou_loss]
    #     loss_weights = [0.5, 0.5]
    #
    #     self.losses = nn.ModuleList(losses)
    #     self.loss_weights = loss_weights
    #     self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
    #     self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5
    #
    #     self.kd_loss = DIST()
    #     # self.kd_loss.cuda()
    #     self.lambda_kd = cfg.TRAIN_LAMBDA_KD

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, skip_long_term_update=False):

        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v, curr_id_k, curr_id_v = lstt_curr_memories[
                layer_idx]
            curr_id_k, curr_id_v = self.AOT.LSTT.layers[
                layer_idx].fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb)
            lstt_curr_memories[layer_idx][2], lstt_curr_memories[layer_idx][
                3] = curr_id_k, curr_id_v
            local_curr_id_k = seq_to_2d(
                curr_id_k, self.enc_size_2d) if curr_id_k is not None else None
            local_curr_id_v = seq_to_2d(curr_id_v, self.enc_size_2d)
            lstt_curr_memories_2d.append([
                seq_to_2d(curr_k, self.enc_size_2d),
                seq_to_2d(curr_v, self.enc_size_2d), local_curr_id_k,
                local_curr_id_v
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            # skip the update of long-term memory or not
            if not skip_long_term_update: 
                self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step

    # def calculate_current_loss(self, gt_mask, step):
    #     pred_id_logits = self.pred_id_logits
    #     pred_id_logits = F.interpolate(pred_id_logits,
    #                                    size=gt_mask.size()[-2:],
    #                                    mode="bilinear",
    #                                    align_corners=self.align_corners)
    #
    #     label_list = []
    #     logit_list = []
    #
    #     for batch_idx, obj_num in enumerate(self.obj_nums):
    #         now_label = gt_mask[batch_idx].long()
    #         now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
    #         label_list.append(now_label.long())
    #         logit_list.append(now_logit)
    #
    #     kd_loss = 0
    #     if self.t_pred_id_logits!=None:
    #         t_pred_id_logit = self.t_pred_id_logits[self.frame_step]
    #         print("=======", len(t_pred_id_logit), len(gt_mask))
    #         for batch_idx, obj_num in enumerate(self.obj_nums):
    #             now_t_label = t_pred_id_logit[batch_idx].long()
    #             now_s_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
    #             # label_list.append(now_t_label.long())
    #             # logit_list.append(now_s_logit)
    #             kd_loss += self.lambda_kd * self.criterion_kd(now_t_label, now_s_logit)
    #
    #     total_loss = kd_loss
    #     for loss, loss_weight in zip(self.losses, self.loss_weights):
    #         total_loss = total_loss + loss_weight * \
    #             loss(logit_list, label_list, step)
    #
    #     return total_loss

class KD_DeAOTInferEngine(AOTInferEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = KD_DeAOTEngine(self.AOT, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            aot_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[separated_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)

            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()
