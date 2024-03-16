# 基於該配置進行繼承並重寫部分配置
_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

data_root = './data/motorcycle_one/'  # 數據集根路徑
class_name = ('humanridebike', )  # 數據集類別名稱

num_classes = len(class_name)  # 數據集類別數
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

anchors = [                          # 多尺度的先驗框基本尺寸
    [(22, 41), (32, 54), (42, 72)],
    [(55, 58), (62, 84), (72, 116)],
    [(100, 115), (113, 154), (154, 143)]
]

base_lr = 0.001
max_epochs = 50

train_batch_size_per_gpu = 32
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/test/')))

_base_.optim_wrapper.optimizer.lr = base_lr
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend',init_kwargs={"project": "motorcycle_onecategory"})]) # noqa
