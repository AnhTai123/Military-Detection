# =====================================================================================
# GroundingDINO Swin-B fine-tune Military v2
# -------------------------------------------------------------------------------------
# Kế thừa từ config cũ, CHỈ override các phần cải thiện. KHÔNG sửa config gốc.
#
# Mục tiêu cải thiện (theo phân tích test):
#   - Tăng recall/AP cho M1 Abrams & Bradley (class yếu nhất)
#   - Giảm nhầm F-22 <-> F-35
#   - Giảm false positive Gorshkov
#   - Giảm overfit valant (valid tốt nhưng test thấp): LR nhỏ hơn + early best ckpt
#
# Các thay đổi chính so với v1:
#   1. Train data: dùng train json đã oversample class yếu (tạo bằng
#      make_oversampled_train.py). Có fallback nếu chưa tạo: dùng RepeatDataset.
#   2. Augmentation multi-scale vừa phải + RandomFlip (không crop mạnh -> giữ object).
#   3. LR nhỏ hơn (overfit): base lr giảm còn ~ 1/2 -> 5e-5 -> thử 1e-5.
#   4. Backbone LR multiplier thấp (giữ pretrained, ít phá feature).
#   5. max_epochs vừa phải + save best theo coco/bbox_mAP + log đầy đủ.
#   6. Đảm bảo metainfo classes khớp TUYỆT ĐỐI với COCO categories (full name).
# =====================================================================================

# Kế thừa config cũ. Sửa đường dẫn này nếu tên file gốc khác.
_base_ = './grounding_dino_swin-b_finetune_military.py'

# ── đường dẫn dataset ────────────────────────────────────────────────────────────
data_root = '/home/aiplatform/workspace/research/research_res/merged_dataset/'
img_prefix = 'images/'

# Train json oversampled (tạo bằng tools/make_oversampled_train.py).
# Nếu CHƯA tạo file này, đổi về 'labels/train_gdino.json' và bật RepeatDataset bên dưới.
train_ann = 'labels/train_gdino_oversampled.json'
val_ann = 'labels/val_gdino.json'
test_ann = 'labels/test_gdino.json'

# ── classes: PHẢI khớp tuyệt đối COCO categories (full name) ─────────────────────
class_names = (
    'Zumwalt class destroyer',
    'Admiral Gorshkov class frigate',
    'F-22 Raptor fighter jet',
    'F-35 Lightning II fighter jet',
    'BM-30 Smerch multiple rocket launcher',
    'M2 Bradley infantry fighting vehicle',
    'BTR-90 armored personnel carrier',
    'HIMARS rocket artillery launcher',
    'M1 Abrams main battle tank',
    'T-90 main battle tank',
)
num_classes = len(class_names)
metainfo = dict(classes=class_names)

# ── model: giữ nguyên kiến trúc, chỉ chỉnh num_classes cho chắc ───────────────────
model = dict(
    bbox_head=dict(num_classes=num_classes),
    # Test config: tăng max_per_img nếu ảnh có nhiều object; nới score thr để
    # tăng recall (lọc sau bằng score_thr lúc eval).
    test_cfg=dict(max_per_img=300),
)

# ── augmentation pipeline ─────────────────────────────────────────────────────────
# Multi-scale resize vừa phải, KHÔNG random crop mạnh (giữ object quân sự nguyên vẹn).
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    # multi-scale: giữ cạnh dài hợp lý cho máy bay/tàu/xe
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                # nhánh resize-crop NHẸ (crop tối thiểu, tránh mất object)
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),  # KHÔNG cho crop mất hết object
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # RandomSamplingNegPos BỊ BỎ: transform này kỳ vọng text là dict (GLIP caption
    # format), nhưng khi dùng COCO dataset + return_classes=True thì text là tuple
    # tên class → gây AttributeError: 'tuple' object has no attribute 'items'.
    # Base config đã xử lý text đúng cách qua dataset pipeline, không cần override.
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

# ── dataloaders ───────────────────────────────────────────────────────────────────
train_dataloader = dict(
    dataset=dict(
        # nếu DÙNG train json oversampled thì repeat_time=1.
        # nếu KHÔNG oversample, đổi ann_file='labels/train_gdino.json' và
        # bọc bằng RepeatDataset (xem ghi chú cuối file).
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        pipeline=train_pipeline,
        return_classes=True,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True,
    ),
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=dict(img=img_prefix),
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True,
    ),
)

# ── evaluators: classwise=True để in bảng per-class AP ───────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann,
    metric='bbox',
    classwise=True,
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + test_ann,
    metric='bbox',
    classwise=True,
    format_only=False,
)

# ── optimizer: LR nhỏ hơn để giảm overfit, backbone LR thấp ──────────────────────
# v1 thường base lr ~ 1e-4 (hoặc 2e-4). Ở đây hạ xuống 5e-5; có thể thử 1e-5.
base_lr = 5e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # giữ pretrained backbone -> LR rất nhỏ
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            # text encoder (BERT) cũng giữ nhẹ để không phá prompt embedding
            'language_model': dict(lr_mult=0.1),
        }))

# ── schedule: train vừa phải + warmup + cosine, save best ────────────────────────
max_epochs = 30
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg_loop = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=500),  # warmup 500 iter
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 26],
        gamma=0.1),
]

# ── hooks: log đầy đủ + lưu best checkpoint theo val mAP ──────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50),
)

# log train/val loss + mAP qua epoch (tensorboard)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Early stopping (tùy chọn): dừng nếu val mAP không cải thiện 5 epoch.
# Bỏ comment nếu muốn dùng.
# custom_hooks = [
#     dict(
#         type='EarlyStoppingHook',
#         monitor='coco/bbox_mAP',
#         rule='greater',
#         patience=5,
#         min_delta=0.001),
# ]

# ── load checkpoint cũ làm điểm khởi đầu (warm start) ─────────────────────────────
# Tiếp tục từ best checkpoint v1 để không train lại từ đầu. Đặt None nếu muốn
# train từ pretrained gốc.
load_from = '/home/aiplatform/workspace/mmdetection/work_dirs/gdino_military_swinb/best_coco_bbox_mAP_epoch_17.pth'

# =====================================================================================
# GHI CHÚ — nếu KHÔNG dùng train json oversampled, thay train_dataloader.dataset bằng:
#
# train_dataloader = dict(
#     dataset=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             type='CocoDataset',          # hoặc type dataset gốc trong _base_
#             data_root=data_root,
#             ann_file='labels/train_gdino.json',
#             data_prefix=dict(img=img_prefix),
#             metainfo=metainfo,
#             pipeline=train_pipeline,
#             return_classes=True,
#             filter_cfg=dict(filter_empty_gt=False),
#         )))
#
# Lưu ý: RepeatDataset lặp TOÀN BỘ dataset, không lặp riêng class yếu. Để oversample
# riêng class yếu, ưu tiên dùng make_oversampled_train.py (đã viết sẵn).
# =====================================================================================
