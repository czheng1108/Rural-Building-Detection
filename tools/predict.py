import os
import sys
sys.path.append(os.getcwd())
import time
import json
import torch
from PIL import Image
# from PIL import ImageFile
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files.retinanet import RetinaNet
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from utils.predict_utils.draw_box_utils import draw_box
import utils.predict_utils.write_xml as wxml
import config.configs as cfg


def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            if (file_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg',
                                            '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                list_name.append(file_path)


def filter_low_thresh(boxes, classes, scores, thresh=0.5):
    return boxes[scores > thresh], classes[scores > thresh], scores[scores > thresh]


def save_image(original_img, predict_boxes, predict_classes, predict_scores, category_index, image_save_path):
    draw_box(original_img,
             predict_boxes,
             predict_classes,
             predict_scores,
             category_index,
             thresh=0.5,
             line_thickness=3)
    if cfg.predict['plot_image']:
        plt.imshow(original_img)
        plt.show()
    # 保存预测的图片结果
    original_img.save(image_save_path)


def write_xml(predict_boxes, predict_classes, predict_scores, category_index, img_path, img):
    predict_boxes, predict_classes, predict_scores = filter_low_thresh(predict_boxes,
                                                                       predict_classes,
                                                                       predict_scores)
    objects = []
    for i in range(predict_scores.shape[0]):
        bandbox = {}
        object_ = {}
        name = category_index[predict_classes[i]]
        bandbox['xmin'] = str(int(predict_boxes[i][0]))
        bandbox['ymin'] = str(int(predict_boxes[i][1]))
        bandbox['xmax'] = str(int(predict_boxes[i][2]))
        bandbox['ymax'] = str(int(predict_boxes[i][3]))
        scores = str(predict_scores[i])
        object_['name'] = name
        object_['bandbox'] = bandbox
        object_['scores'] = scores
        objects.append(object_)

    save_img_path = img_path.replace(cfg.predict['image_path'], 'test_result/xml_write').split('.')[0] + '.xml'
    # save_img_path = os.path.join('../', save_img_path)
    save_img_dir = save_img_path.replace(save_img_path.split('/')[-1], '')

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    wxml.write_xml(save_path=save_img_path,
                   folder='JPEGImages',
                   filename=img_path.split("/")[-1],
                   # path='./JPEGImages/'+img_path.split('/')[-1],
                   path=img_path,
                   size=dict(width=str(img.shape[2]),
                             height=str(img.shape[3]),
                             depth=str(img.shape[1])),
                   object_list=objects
                   )


def main(image_list, train_weights):
    # get devices
    print("using {} to predict!".format(train_weights.split("/")[-1].split(".")[0]))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=3)

    # load train weights
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    # label_json_path = '../data/VOCdevkit/data_classes.json'
    label_json_path = cfg.classes
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    init = True

    # load image
    for img_path in image_list:
        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Image.MAX_IMAGE_PIXELS = None
        original_img = Image.open(img_path)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            if init:
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)
                init = False

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+{} time: {}".format(cfg.nms['fun'], (t_end - t_start)))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            # save image
            if cfg.predict['save_image']:
                image_save_path = img_path.replace(cfg.predict['image_path'], 'test_result/img_write', 1)
                make_path = image_save_path.replace('/' + image_save_path.split('/')[-1], '')
                if not os.path.exists(make_path):
                    os.makedirs(make_path)
                # image_save_path = os.path.join('./test_result/img_write', img_path.split("/")[-1])
                save_image(original_img,
                           predict_boxes,
                           predict_classes,
                           predict_scores,
                           category_index,
                           image_save_path)

            # write xml
            if cfg.predict['save_voc']:
                write_xml(predict_boxes,
                          predict_classes,
                          predict_scores,
                          category_index,
                          img_path, img)


if __name__ == '__main__':
    image_list = []
    listdir(cfg.predict['image_path'], image_list)
    train_weights = cfg.resume
    main(image_list, train_weights)

