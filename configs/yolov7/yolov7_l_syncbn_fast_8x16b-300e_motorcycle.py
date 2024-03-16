# 基於該配置進行繼承並重寫部分配置
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

num_classes = len(class_name)  # 取得類別總數
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])  #

anchors = [
    [(22, 41), (32, 54), (42, 72)],
    [(55, 58), (62, 84), (72, 116)],
    [(100, 115), (113, 154), (154, 143)]
]

base_lr = 0.001  # 適用於所有層的學習率
max_epochs = 50  # 最大訓練50世代

train_batch_size_per_gpu = 32   # 訓練時單個GPU的Batch Size
train_num_workers = 4           # 訓練時單個GPU分配的數據加載線程數

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'  # noqa


# 將所有的 `num_classes` 默認值修改為5（原来为80）
model = dict(
    backbone=dict(frozen_stages=4),   # 固定整個 backbone 權重，不進行訓練
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,      # 表示單次傳遞給程式用以訓練的數據(樣本)個數
    num_workers=train_num_workers,            # 指定在數據加載過程中使用的工作線程數
    dataset=dict(    # Dataset的原始配置信息
        data_root=data_root,
        metainfo=metainfo,  # 包含數據集的元信息，例如類別信息
        ann_file='annotations/train.json',        # 標註文件路徑
        data_prefix=dict(img='images/train/')))   # 圖像路徑前缀

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,  # 包含數據集的元信息，例如類別信息
        data_root=data_root,
        ann_file='annotations/val.json',          # 標註文件路徑
        data_prefix=dict(img='images/val/')))   # 圖像路徑前缀

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,  # 包含數據集的元信息，例如類別信息
        data_root=data_root,
        ann_file='annotations/test.json',         # 標註文件路徑
        data_prefix=dict(img='images/test/')))   # 圖像路徑前缀

_base_.optim_wrapper.optimizer.lr = base_lr
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

default_hooks = dict(
    # 每隔 10 个 epoch 保存一次權重，並且最多保存2個權重
    # 模型評估時候自動保存最佳模型
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    # warmup_mim_iter 參數非常關鍵，因為 cat 數據集非常小，默認的最小 warmup_mim_iter 是 1000，導致訓練過程學習率偏小
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    # 日志打印間隔為 5
    logger=dict(type='LoggerHook', interval=5))
# 評估間隔為 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# 視覺化工具
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend',init_kwargs={"project": "motorcycle"})]) # noqa
