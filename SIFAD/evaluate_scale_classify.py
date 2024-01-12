import cv2
import numpy as np
from PIL import Image
import os

import xml.etree.ElementTree as ET

from tqdm import tqdm
from yolo import YOLO
yolo = YOLO()
VOCdevkit_path  = 'VOCdevkit'
test_set = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test0.txt")).read().strip().split()

correct_num = 0

for img in tqdm(test_set):
    image = Image.open(os.path.join('VOCdevkit/VOC2007/JPEGImages', img+'.jpg'))
    scale_cls_pre = yolo.detect_scale(image)
    # print(scale_cls_pre)
    in_file = os.path.join('VOCdevkit/VOC2007/Annotations', img+'.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()
    scale_cls = int(root.find('scale_class').text)
    # print(scale_cls)
    if scale_cls == scale_cls_pre:
        correct_num += 1

precision = correct_num/len(test_set)
print(precision)
"""
--------------------
30625
24698
class:4
--------------------
24693
18418
class:3
--------------------
18412
13828
class:2
--------------------
13821
9180
class:1
--------------------
9153 
2806
class:0
--------------------
"""
