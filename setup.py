from setuptools import setup
from Cython.Build import cythonize
import shutil
import os

# compile backbone
setup(ext_modules=cythonize('backbone/attention.pyx'))
setup(ext_modules=cythonize('backbone/feature_pyramid_network.pyx'))
setup(ext_modules=cythonize('backbone/resnet50_fpn_model.pyx'))
for file in os.listdir('./'):
    if file.endswith('.so'):
        shutil.move(file, 'backbone/')


# compile network_files
setup(ext_modules=cythonize('network_files/anchor_utils.pyx'))
setup(ext_modules=cythonize('network_files/boxes.pyx'))
setup(ext_modules=cythonize('network_files/det_utils.pyx'))
setup(ext_modules=cythonize('network_files/image_list.pyx'))
setup(ext_modules=cythonize('network_files/retinanet.pyx'))
setup(ext_modules=cythonize('network_files/transform.pyx'))
for file in os.listdir('./'):
    if file.endswith('.so'):
        shutil.move(file, 'network_files/')


# compile network_files/loss
setup(ext_modules=cythonize('network_files/loss/gfocal_loss.pyx'))
setup(ext_modules=cythonize('network_files/loss/ghm_loss.pyx'))
setup(ext_modules=cythonize('network_files/loss/losses.pyx'))
setup(ext_modules=cythonize('network_files/loss/utils.pyx'))
for file in os.listdir('./'):
    if file.endswith('.so'):
        shutil.move(file, 'network_files/')


# compile network_files/nms
setup(ext_modules=cythonize('network_files/nms/diou_nms.pyx'))
setup(ext_modules=cythonize('network_files/nms/soft_nms.pyx'))
for file in os.listdir('./'):
    if file.endswith('.so'):
        shutil.move(file, 'network_files/')