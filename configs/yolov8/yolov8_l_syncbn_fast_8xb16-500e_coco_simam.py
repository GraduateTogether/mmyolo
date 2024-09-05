_base_ = './yolov8_l_syncbn_fast_8xb16-500e_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(cfg=dict(type='SimAM'),
                 stages=(False, False, True, True))
        ]
    )
)
