config = dict(
    learning_rate=3e-4,
    weight_decay=1e-4,
    batch_size=16,
    epoch=200,
    clip=(-1024.0, 512.0),
    mean=-236.88525,
    std=404.0286,
    shape_train=(256, 224, 288),
    path_data_train='/raid/wangqi/data_npz_new',
    path_csv='/raid/wangqi/code/train_model/preparation/information.csv',
    path_save='/raid/wangqi/code/train_model/experiments/severe_img_age/split0',
    path_pretrain='/raid/wangqi/code/train_model/pretrain/resnet_18_23dataset.pth',
    num_classes=2,
    val_frequency=5,
    val_split=0
)
