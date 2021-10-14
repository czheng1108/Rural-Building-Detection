import os
import sys
sys.path.append(os.getcwd())
import datetime
import torch
from utils import transforms
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files.retinanet import RetinaNet
from utils.my_dataset import VOC2012DataSet
from utils.train_utils import train_eval_utils as utils
import config.configs as cfg
import utils.log_utils.check_file as check_file
# from torch.utils.tensorboard import SummaryWriter


def create_model(num_classes, device):
    # skip P2 because it generates too many anchors (according to their paper)
    # the backbone here uses FrozenBatchNorm2d by default
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3)
    model = RetinaNet(backbone, num_classes)

    # load pre-training weights
    # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
    weights_path = cfg.resume
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    if weights_path.split('/')[-1] != 'retinanet_resnet50_fpn.pth':
        model2dict = model.state_dict()
        model2dict.update(weights_dict['model'])
        model.load_state_dict(model2dict)
        return model

    # remove the weight of the classifier part
    del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
    for k in del_keys:
        del weights_dict[k]
    # print(model.load_state_dict(weights_dict, strict=False))
    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))
    # if not os.path.exists('./log/results_file'):
    #     os.makedirs('./log/results_file')
    results_file = "./log/results_file/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # tensorboard
    # tb_writer = SummaryWriter(log_dir='log/tensorboard/retinaNet_spp')

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    # if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
    if os.path.exists(cfg.data_root) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], "train.txt")

    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    model = create_model(num_classes=parser_data.num_classes + 1, device=device)

    # print(model)
    model.to(device)

    # write model to tensorboard
    # init_img = torch.zeros((1, 3, 1000, 1000), device=device)
    # tb_writer.add_image('image', init_img.view(3, 1000, 1000), 0)
    # tb_writer.add_graph(model, init_img)
    # tb_writer.close()

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr,
                                momentum=0.9, weight_decay=0.005)
    # optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.step_size,
                                                   gamma=cfg.lrs_gamma)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / parser_data.epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # whether to continue training after the last result
    if cfg.continue_training:
        checkpoint = torch.load(parser_data.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    # lr_scheduler.last_epoch = parser_data.start_epoch

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=cfg.print_freq, warmup=cfg.warmup)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test_image dataset
        coco_info, voc = utils.evaluate(model, val_data_set_loader, device=device)

        # add loss, acc, lr into tensorboard
        # tags = ['train_loss', 'accuracy', 'learning_rate']
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

        # write into txt
        with open(results_file, "a") as f:
            # the written data includes coco indicators as well as loss and learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item(), lr]]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            txt = txt + voc
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal map

        # save weights
        if epoch % cfg.checkpoint_config == 0 or epoch == cfg.epochs-1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from utils.log_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from utils.log_utils.plot_curve import plot_map
        plot_map(val_map)

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)
    # print(predictions)
    # tb_writer.close()


def parser_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default=torch.device(cfg.device if torch.cuda.is_available() else 'cpu'),
                        help='device')
    parser.add_argument('--data-path', default=cfg.data_path, help='dataset')
    # number of object categories (excluding background)
    parser.add_argument('--num-classes', default=cfg.num_classes, type=int, help='num_classes')
    parser.add_argument('--output-dir', default=cfg.output_dir, help='path where to save')
    parser.add_argument('--resume', default=cfg.resume, type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # total number of epochs trained
    parser.add_argument('--epochs', default=cfg.epochs, type=int, metavar='N',
                        help='number of total epochs to run')
    # the batch size of training
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int, metavar='N',
                        help='batch size when training.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    check_file.check_dir(args)
    # lg.terminal_log()
    print(args)
    main(args)
