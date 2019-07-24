from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np 
import cv2
DATA_PATH = 'D:\Code\Deep learning\CenterNet\data\widerface'     #give the file path(could modify here)
out_dir = 'D:\Code\Deep learning\CenterNet\data\widerface\\annotations'
labels = ['test']       # the test labels is different and we could change it

import os
import linecache
import re
from PIL import Image
from collections import defaultdict     

# this function is use to count the lines
def count_lines(file):
    lines_quantity = 0
    while True:
        buffer = file.read(1024 * 8192)
        if not buffer:
            break
        lines_quantity += buffer.count('\n')
    file.close()
    return lines_quantity
#use to find the key by value(in the list)
def get_key(dict, value):
    for k, v in dict.items():
        val_len = len(v)
        if value in v:
            return k

def convert_wider2coco(DATA_PATH, out_dir):
    """Now we will  transform the wider_face format to coco format"""

    json_name = 'wf2coco_test.json'    #we could change the .json filename
    img_id = 0
    ann_id = 0
    cat_id = 1

    for label in labels:    #loop for labels

        print("Transforming....")
        ann_dict = {}
        categories = [{"id": 1, "name": 'face'}]
        images = []
        annotations = []
        ann_file = DATA_PATH + '\wider_face_split' + '\wider_face_{}_filelist.txt'.format(label)
        wider_dict = []
        file1 = open(ann_file, 'r')
        lines = count_lines(file1)
        #fold_dict = defaultdict(list)

        #with open("file") as fh:
        #    for line in fh:
        #        if len(line) > 10:  #widerface's file name is longer than img_info
        #            filename_list.append(line)  #create a list with widerface_key
        #        else:
        #            pass
        
        #we choose the regular express
        for i in range(lines):
            line = linecache.getline(ann_file, i+1)

            image = {}
            image['id'] = img_id
            img_id += 1
            data_dir = DATA_PATH + '\WIDER_{}'.format(label) + '\images\\'
            position = line.index('/')
            file_name = line[position + 1: -5]
            folder_name = line[:position]
            filname = folder_name + '\\' + file_name + '.jpg'
            im = Image.open(data_dir + filname)
            image['width'] = im.width
            image['height'] = im.height
            image['file_name'] = file_name + '.jpg'
            images.append(image)

    ann_dict['images'] = images
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))

convert_wider2coco(DATA_PATH, out_dir)