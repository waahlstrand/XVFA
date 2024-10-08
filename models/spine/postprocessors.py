import torch
import torch.nn as nn
import models.backbones.DINO.util.box_ops as box_ops
from torchvision.ops.boxes import nms

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False, as_records=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        # Cheat for the matcher in evaluation, pred_logits is expected
        # but we return the one-hot labels
        pred_logits = torch.zeros((out_logits.shape[0], num_select, out_logits.shape[2]), device=out_logits.device)
        pred_logits.scatter_(2, labels.unsqueeze(-1), 1)

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if not as_records:

            if self.nms_iou_threshold > 0:

                item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
                scores = torch.stack([s[i] for s, i in zip(scores, item_indices)])
                labels = torch.stack([l[i] for l, i in zip(labels, item_indices)])
                boxes = torch.stack([b[i] for b, i in zip(boxes, item_indices)])
                pred_logits = torch.stack([l[i] for l, i in zip(pred_logits, item_indices)])

            results = {
                'scores': scores,
                'labels': labels,
                'boxes': boxes,
            }

            # Compatibility with matcher
            results['pred_logits'] = pred_logits
            results['pred_boxes'] = boxes

        else:
            if self.nms_iou_threshold > 0:
                item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

                results = [
                    {
                        'scores': s[i], 
                        'labels': l[i], 
                        'boxes': b[i], 
                        'pred_boxes': b[i], 
                        'pred_logits': p[i]
                    } 
                        for s, l, b, p, i in zip(scores, labels, boxes, pred_logits, item_indices)
                ]
            else:

                results = [{'scores': s, 'labels': l, 'boxes': b, 'pred_boxes': b, 'pred_logits': p} for s, l, b, p in zip(scores, labels, boxes, pred_logits)]

        return results