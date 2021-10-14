from flashtorch.activmax import GradientAscent
import matplotlib.pyplot as plt
import torch
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files import RetinaNet
import os


def create_model(num_classes, device):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)
    # 载入预训练模型权重
    weights_path = './save_weights/model-5.pth'
    # weights_path = '../pre_training_weights/model30-spp-1-1.pth'
    weights_path = '../save_weights/resNetFpn-model-1.pth'
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model'])
    return model


def show_architecture(architecture):
    for i in range(len(architecture)):
        print(architecture[i])


if __name__ == '__main__':
    model = create_model(2+1, 'cuda:0')
    model = model.backbone
    architecture = list(model.body.named_children())  # Print layers and corresponding indicies
    show_architecture(architecture)
    g_ascent = GradientAscent(model.body)
    conv = model.body.layer4[2].conv3
    # show filter
    # conv1_2_filters = [17, 33, 34, 57]
    # output = g_ascent.visualize(conv, conv1_2_filters, title='conv1_2', return_output=True)
    # plt.show()
    if not os.path.exists('../test_result/feature_map'):
        os.makedirs('../test_result/feature_map')
    for i in range(0, 500):
        g_ascent.deepdream('../test_image/dbc-2x_2-266.png', conv, i, title='m{}'.format(i))
        plt.savefig('../test_result/feature_map/img{}.png'.format(i), dpi=400, bbox_inches='tight')
        print('saving img{}.png!'.format(i))

