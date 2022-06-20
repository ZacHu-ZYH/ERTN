# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import numpy as np
import cv2
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print(samples,targets)
        outputs = model(samples)
        # print(outputs['pred_masks'].shape)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    cnt=-1
    for batch_idx, (samples, targets,image_id) in enumerate(data_loader):
        print(image_id[0][:-4])
        ##############for tumor##########
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # compute the scores, excluding the "no-object" class (the last one)
        scores = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
        # threshold the confidence
        keep = scores > 0
        for i,gt in enumerate(targets):
            img_h, img_w = gt["size"][0], gt["size"][1]
            tt = gt["orig_size"]

        for i,mask in enumerate(outputs["pred_masks"][keep]):
    

            
            if i==0:
                buwei = "tc"
            elif i==1:
                buwei = "wt"
            elif i==2:
                buwei = "et"
            final_mask = mask[:img_h, :img_w]
            # print('ddd',final_mask.shape)
            final_mask = cv2.resize(np.array(final_mask.cpu()),((np.array(tt.cpu()[0]),np.array(tt.cpu()[1]))))
            final_mask = np.clip(final_mask,0,1)
            # print(final_mask)
            # final_mask = F.interpolate(
            #     final_mask.float(), size=tuple(tt.tolist()), mode="nearest"
            # ).byte()

            # cnts = contours[0] if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是
            # plt.imshow(final_mask,"gray")

            # for cnt in cnts:
            # final_mask = np.array(final_mask,dtype=np.uint8)
            # box = cv2.boundingRect(final_mask)
            # final_mask = final_mask[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2])]
            cv2.imwrite('E:\HuZhaoyu\ERTN\output\output_images_tumor\%s_et.jpg'%(image_id[0][:-4]),final_mask*255)
        #########
        
        # print(batch_idx)
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # outputs = model(samples)
        # # compute the scores, excluding the "no-object" class (the last one)
        # scores = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
        # # threshold the confidence
        # keep = scores > 0
        # for i,(mask,gt) in enumerate(zip(outputs["pred_masks"][keep], targets)):
        #     img_h, img_w = gt["size"][0], gt["size"][1]
        #     tt = gt["orig_size"]

            
        #     final_mask = mask[:img_h, :img_w]
        #     # print('ddd',final_mask.shape)
        #     final_mask = cv2.resize(np.array(final_mask.cpu()),((np.array(tt.cpu()[0]),np.array(tt.cpu()[1]))))
        #     final_mask = np.clip(final_mask,0,1)
        #     # print(final_mask)
        #     # final_mask = F.interpolate(
        #     #     final_mask.float(), size=tuple(tt.tolist()), mode="nearest"
        #     # ).byte()

        #     # cnts = contours[0] if imutils.is_cv2() else contours[1]  #用imutils来判断是opencv是2还是
        #     # plt.imshow(final_mask,"gray")

        #     # for cnt in cnts:
        #     # final_mask = np.array(final_mask,dtype=np.uint8)
        #     # box = cv2.boundingRect(final_mask)
        #     # final_mask = final_mask[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2])]
        #     cv2.imwrite('E:\HuZhaoyu\ERTN\output\output_images\%s.jpg'%(image_id[0][:-4]),final_mask*255)
            
        #     # plt.show()
            
            
            
            
            
            
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        # result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # # print(orig_target_sizes)
        # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        # results = postprocessors['segm'](targets, outputs, orig_target_sizes, target_sizes)
        #     # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # for i,mask in enumerate(results[1]["masks"]):
                
        #     plt.imshow(np.squeeze(mask.cpu()),'gray')
        #     plt.show()
            
        # if coco_evaluator is not None:
            # coco_evaluator.update(res)


    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]
    return coco_evaluator
