from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np 
import cv2
DATA_PATH = 'D:\Code\Deep learning\CenterNet\data\widerface'     #give the file path(could modify here)
out_dir = 'D:\Code\Deep learning\CenterNet\data\widerface\\annotations'
labels = ['train']       # the test labels is different and we could change it

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

    json_name = 'wf2coco_train.json'    #we could change the .json filename
    img_id = 0
    ann_id = 0
    cat_id = 1

    for label in labels:    #loop for labels

        print("Transforming....")
        ann_dict = {}
        categories = [{"id": 1, "name": 'face'}]
        images = []
        annotations = []
        ann_file = DATA_PATH + '\wider_face_split' + '\wider_face_{}_bbx_gt.txt'.format(label)
        wider_dict = {}
        file1 = open(ann_file, 'r')
        lines = count_lines(file1)
        fold_dict = defaultdict(list)

        #with open("file") as fh:
        #    for line in fh:
        #        if len(line) > 10:  #widerface's file name is longer than img_info
        #            filename_list.append(line)  #create a list with widerface_key
        #        else:
        #            pass
        
        #we choose the regular express
        for i in range(lines):
            line = linecache.getline(ann_file, i)
            #we locate the .jpg line and then seperate the string and then append them to the dict
            if re.search('jpg', line):
                position = line.index('/')
                file_name = line[position + 1: -5]
                folder_name = line[:position]
                i += 1  #Let the line counter move down for one unit
                face_count = int(linecache.getline(ann_file, i))    #this is the face_number in the annotation file
                fold_dict[folder_name].append(file_name)

                for j in range(face_count):
                    box_line = linecache.getline(ann_file, i+j+1)   #locate the box_line
                    po_x1 = box_line.index(' ')
                    x1 = float(box_line[:po_x1])
                    #move the cursor backward for one unit and then locate the " "
                    #but we have to left the space to split them and the space is on their front
                    po_y1 = box_line.index(' ', po_x1 + 1)    
                    y1 = float(box_line[po_x1:po_y1])
                    po_w = box_line.index(' ', po_y1 + 1)
                    w = float(box_line[po_y1:po_w])
                    po_h = box_line.index(' ', po_w + 1)
                    h = float(box_line[po_w:po_h])
                    coordinates = [x1, y1, w, h]
                    #we create a dict(key is file_name and the value is the coordinates)
                    #im_file = [[x, y, w, h], ....]
                    wider_dict.setdefault(file_name, []).append(coordinates)


        #data_dir = DATA_PATH + '\WIDER_{}'.format(label) + '\images'      
        # we need to go to the image folder to get some parameters

        for filename in wider_dict.keys():
            

            image = {}
            image['id'] = img_id
            img_id += 1

            #use the regular expression to map the string
            #pattern = re.compile(r'[0-9]+\w?[a-zA-Z]+')     #give the pattern
            #match1 = re.search(pattern, filename).group(0)     #extra the string
            #position1 = match1.index('_')   #locate the "_" position
            #folder_f = match1[:position1]
            #folder_b = match1[position1+1:]
            #im_folder = folder_f + "--" + folder_b +"\\"  #combine them together
            im_folder = get_key(fold_dict, filename)
            #print(im_folder)
            data_dir = DATA_PATH + '\WIDER_{}'.format(label) + '\images\\' + im_folder + '\\'
            #im = Image.open(os.path.join(data_dir, filename))
            im = Image.open(data_dir + filename + ".jpg")
            image['width'] = im.width
            image['height'] = im.height
            image['file_name'] = filename + '.jpg'
            images.append(image)
            
            #we need the file path to get the gt_bbx
            #we have used the split.py to generate the face_bbx
            bbox_path = DATA_PATH + '\wider_face_split' + '\wider_face_{}'.format(label)

            for gt_bbox in wider_dict[filename]:
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id # 1:"face" for widerface
                ann['iscrowd'] = 0
                ann['area'] = gt_bbox[2] * gt_bbox[3]
                ann['bbox'] = gt_bbox
                annotations.append(ann)

    ann_dict['images'] = images
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))

convert_wider2coco(DATA_PATH, out_dir)