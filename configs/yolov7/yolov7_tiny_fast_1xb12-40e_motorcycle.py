_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

data_root = './data/motorcycle/'
class_name = ('DNoHelmetP1NoHelmet',
'DHelmetP1Helmet',
'DHelmet',
'DNoHelmet',
'DHelmetP1NoHelmet',
'DHelmetP0NoHelmetP1NoHelmet',
'DHelmetP1NoHelmetP2NoHelmet',
'DNoHelmetP1NoHelmetP2NoHelmet',
'DHelmetP1NoHelmetP2Helmet',
'DNoHelmetP1Helmet',
'DHelmetP0NoHelmetP1NoHelmetP2Helmet',
'DNoHelmetP0NoHelmetP1NoHelmet',
'DNoHelmetP0NoHelmet',
'DHelmetP0NoHelmet',
'DNoHelmetP1HelmetP2Helmet',
'DHelmetP1HelmetP2Helmet',
'DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet',
'DHelmetP0NoHelmetP1NoHelmetP2NoHelmet',
'DHelmetP0NoHelmetP1Helmet',
'DHelmetP1HelmetP2NoHelmet',
'DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet',
'DHelmetP0Helmet',
'DNoHelmetP1NoHelmetP2Helmet',
'DHelmetP0NoHelmetP1HelmetP2Helmet',
'DHelmetP1NoHelmetP2NoHelmetP3Helmet',
'DHelmetP0HelmetP1Helmet',
'DNoHelmetP0NoHelmetP1Helmet',
'DHelmetP1NoHelmetP2NoHelmetP3NoHelmet',
'DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet',
'DHelmetP0HelmetP1NoHelmetP2Helmet',
'DHelmetP0HelmetP1NoHelmetP2NoHelmet',
'DNoHelmetP0HelmetP1NoHelmet',
'DHelmetP0HelmetP1HelmetP2Helmet',
'DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet',
'DNoHelmetP0NoHelmetP1NoHelmetP2Helmet',
'DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet',)

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth'  # noqa

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
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
