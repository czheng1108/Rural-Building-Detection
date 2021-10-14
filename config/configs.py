# training parameter
"""
    * lr: Learning rate. default=0.005
    * step_size: Number of steps to decrease learning rate.
    * lrs_gamma: Number of Times to decrease learning rate.
    * device: Training equipment type, best to use cuda.
    * num_classes: Number of detection object categories (excluding background).
    * continue_training: Whether to continue training.
    * warmup: Whether to start warmup training mode.
    * epoch: Number of training epochs.
    * batch_size: Number of training batch size.
    * checkpoint_config: How many epochs to train to save the weight once.
    * classes: Category file path.
    * data_root: Datasets path.
    * resume: If you need to continue with the last training, specify the address of the last training save weight file.
    * output_dir: Training result save address.
    * data_path: The root directory of the training data set (VOCdevkit).
    * print_freq: How many batches to print training information once.
"""
# optimizer parameter0.005
lr = 0.005
# learning rate scheduler
step_size = 3
lrs_gamma = 0.33
# other parameter
device = 'cuda:0'  # 训练设备类型
num_classes = 2  # 检测目标类别数(不包含背景)
continue_training = False  # 是否继续训练
warmup = True  # 启用warmup训练方式，可理解为热身训练
epochs = 2    # 训练的总epoch数
batch_size = 15  # 训练的batch size
checkpoint_config = 100  # 训练多少轮次保存一次权重
# path
classes = './data/VOCdevkit/data_classes.json'  # 类别文件路径
data_root = './data/VOCdevkit/VOC2012'  # 数据集路径
resume = './pre_training_weights/model50-attention-spp-1-3.pth'
# resume = 'pre_training_weights/retinanet_resnet50_fpn.pth'
# resume = './save_weights/resNetFpn-model-14.pth'
output_dir = './save_weights'  # 文件保存地址
data_path = './data' # 训练数据集的根目录(VOCdevkit)
# show training information
print_freq = 20  # 多少批次打印一次训练信息

# Non-Maximum Suppression
"""
    Include three post-processing algorithms: nms, batched_nms, diou_nms, soft_nms
    Recommended parameters:
    * nms['thresh']=0.60
    * batched_nms['thresh']=0.60
    * diou_nms['thresh']=0.55
    * soft_nms['thresh']=0.25
"""
nms = dict(fun='batched_nms',
           thresh=0.50)

# loss function
"""
    Include classification loss and bounding box regression loss.
    * Classification loss include focal_loss and ghm_loss.
    * Box regression loss include l1_loss, smooth_l1_loss, Iou_loss, Giou_loss, Diou_loss and Ciou_loss.
    Parameter Description:
    * fun: Loss function name
    * reduction: The methon to calculate loss.
    * alpha: Focal_loss parameter. If you use ghm loss, please ignore this parameter.
    * gamma: Focal_loss parameter. If you use ghm loss, please ignore this parameter.
    * coefficient: Used to balance loss.  
"""
cls_loss = dict(fun='focal_loss',
                reduction="sum",
                alpha=0.65,  # default=0.65
                gamma=2,  # default=2
                coefficient=1.5) # default=1.5

reg_loss = dict(fun='Ciou_loss',
                reduction='sum',
                coefficient=0.65) # default=0.65

# attention
"""
    * fun: include 'Attentions', 'None'
    * site: The position of the attention module, where 1, 2, and 3 correspond to p1, p2, and p3 in our paper.
"""
attention = dict(fun='Attention',
                 site=[1, 2, 3])

# spatial pyramid network for feature fusion
spn = True

# predict
"""
    * image_path: The path of the image to be detected.
    * save_image: Whether to save the prediction results as an image.
    * plot_image: Whether to draw prediction results.
    * save_voc: Whether to save the prediction result as a voc format xml file.
"""
predict = dict(image_path='test_image',
               save_image=True,
               plot_image=False,
               save_voc=False)

