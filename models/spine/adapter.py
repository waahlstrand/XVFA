import lightning as L
from typing import *
from models.base import Detector


class DINO(Detector):
    """
    Adapter class for the DINO arguments, to ensure
    compatibility with the LightningCLI.
    
    """

    def __init__(self, 
                 n_keypoints: int,
                 n_vertebra: int,
                 n_dims: int,
                 missing_weight: float,
                 checkpoint: str,
                 num_classes: int,
                 num_queries: int,
                 num_select: int,
                 lr: float,
                 param_dict_type: str,
                 lr_backbone: float,
                 lr_backbone_names: List[str],
                 lr_linear_proj_names: List[str],
                 lr_linear_proj_mult: float,
                 ddetr_lr_param: bool,
                 batch_size: int,
                 weight_decay: float,
                 epochs: int,
                 lr_drop: int,
                 save_checkpoint_interval: int,
                 clip_max_norm: float,
                 onecyclelr: bool,
                 multi_step_lr: bool,
                 lr_drop_list: List[int],
                 modelname: str,
                 frozen_weights: Optional[str],
                 backbone: str,
                 use_checkpoint: bool,
                 dilation: bool,
                 position_embedding: str,
                 pe_temperatureH: int,
                 pe_temperatureW: int,
                 return_interm_indices: List[int],
                 backbone_freeze_keywords: Optional[str],
                 enc_layers: int,
                 dec_layers: int,
                 unic_layers: int,
                 pre_norm: bool,
                 dim_feedforward: int,
                 hidden_dim: int,
                 dropout: float,
                 nheads: int,
                 query_dim: int,
                 num_patterns: int,
                 pdetr3_bbox_embed_diff_each_layer: bool,
                 pdetr3_refHW: int,
                 random_refpoints_xy: bool,
                 fix_refpoints_hw: int,
                 dabdetr_yolo_like_anchor_update: bool,
                 dabdetr_deformable_encoder: bool,
                 dabdetr_deformable_decoder: bool,
                 use_deformable_box_attn: bool,
                 box_attn_type: str,
                 dec_layer_number: Optional[int],
                 num_feature_levels: int,
                 enc_n_points: int,
                 dec_n_points: int,
                 decoder_layer_noise: bool,
                 dln_xy_noise: float,
                 dln_hw_noise: float,
                 add_channel_attention: bool,
                 add_pos_value: bool,
                 two_stage_type: str,
                 two_stage_pat_embed: int,
                 two_stage_add_query_num: int,
                 two_stage_bbox_embed_share: bool,
                 two_stage_class_embed_share: bool,
                 two_stage_learn_wh: bool,
                 two_stage_default_hw: float,
                 two_stage_keep_all_tokens: bool,
                 transformer_activation: str,
                 batch_norm_type: str,
                 masks: bool,
                 aux_loss: bool,
                 set_cost_class: float,
                 set_cost_bbox: float,
                 set_cost_giou: float,
                 cls_loss_coef: float,
                 mask_loss_coef: float,
                 dice_loss_coef: float,
                 bbox_loss_coef: float,
                 giou_loss_coef: float,
                 polynomial_loss_coef: float,
                 enc_loss_coef: float,
                 interm_loss_coef: float,
                 no_interm_box_loss: bool,
                 focal_alpha: float,
                 decoder_sa_type: str,
                 matcher_type: str,
                 decoder_module_seq: List[str],
                 nms_iou_threshold: float,
                 dec_pred_bbox_embed_share: bool,
                 dec_pred_class_embed_share: bool,
                 use_dn: bool,
                 dn_number: int,
                 dn_box_noise_scale: float,
                 dn_label_noise_ratio: float,
                 embed_init_tgt: bool,
                 dn_labelbook_size: int,
                 match_unstable_error: bool,
                 use_ema: bool,
                 ema_decay: float,
                 ema_epoch: int,
                 use_detached_boxes_dec_out: bool
                 ) -> None:
        super().__init__()


        self.n_keypoints = n_keypoints
        self.n_vertebra = n_vertebra
        self.n_dims = n_dims
        self.missing_weight = missing_weight
        # self.filter = filter

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_select = num_select
        self.lr = lr
        self.param_dict_type = param_dict_type
        self.lr_backbone = lr_backbone
        self.lr_backbone_names = lr_backbone_names
        self.lr_linear_proj_names = lr_linear_proj_names
        self.lr_linear_proj_mult = lr_linear_proj_mult
        self.ddetr_lr_param = ddetr_lr_param
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr_drop = lr_drop
        self.save_checkpoint_interval = save_checkpoint_interval
        self.clip_max_norm = clip_max_norm
        self.onecyclelr = onecyclelr
        self.multi_step_lr = multi_step_lr
        self.lr_drop_list = lr_drop_list
        self.modelname = modelname
        self.frozen_weights = frozen_weights
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint
        self.dilation = dilation
        self.position_embedding = position_embedding
        self.pe_temperatureH = pe_temperatureH
        self.pe_temperatureW = pe_temperatureW
        self.return_interm_indices = return_interm_indices
        self.backbone_freeze_keywords = backbone_freeze_keywords
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.unic_layers = unic_layers
        self.pre_norm = pre_norm
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nheads = nheads
        self.query_dim = query_dim
        self.num_patterns = num_patterns
        self.pdetr3_bbox_embed_diff_each_layer = pdetr3_bbox_embed_diff_each_layer
        self.pdetr3_refHW = pdetr3_refHW
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw
        self.dabdetr_yolo_like_anchor_update = dabdetr_yolo_like_anchor_update
        self.dabdetr_deformable_encoder = dabdetr_deformable_encoder
        self.dabdetr_deformable_decoder = dabdetr_deformable_decoder
        self.use_deformable_box_attn = use_deformable_box_attn
        self.box_attn_type = box_attn_type
        self.dec_layer_number = dec_layer_number
        self.num_feature_levels = num_feature_levels
        self.enc_n_points = enc_n_points
        self.dec_n_points = dec_n_points
        self.decoder_layer_noise = decoder_layer_noise
        self.dln_xy_noise = dln_xy_noise
        self.dln_hw_noise = dln_hw_noise
        self.add_channel_attention = add_channel_attention
        self.add_pos_value = add_pos_value
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share
        self.two_stage_learn_wh = two_stage_learn_wh
        self.two_stage_default_hw = two_stage_default_hw
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.transformer_activation = transformer_activation
        self.batch_norm_type = batch_norm_type
        self.masks = masks
        self.aux_loss = aux_loss
        self.set_cost_class = set_cost_class
        self.set_cost_bbox = set_cost_bbox
        self.set_cost_giou = set_cost_giou
        self.cls_loss_coef = cls_loss_coef
        self.mask_loss_coef = mask_loss_coef
        self.dice_loss_coef = dice_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.polynomial_loss_coef = polynomial_loss_coef
        self.enc_loss_coef = enc_loss_coef
        self.interm_loss_coef = interm_loss_coef
        self.no_interm_box_loss = no_interm_box_loss
        self.focal_alpha = focal_alpha
        self.decoder_sa_type = decoder_sa_type
        self.matcher_type = matcher_type
        self.decoder_module_seq = decoder_module_seq
        self.nms_iou_threshold = nms_iou_threshold
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.use_dn = use_dn
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.embed_init_tgt = embed_init_tgt
        self.dn_labelbook_size = dn_labelbook_size
        self.match_unstable_error = match_unstable_error
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epoch = ema_epoch
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out