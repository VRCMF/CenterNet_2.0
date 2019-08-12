from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np 
import cv2
import os

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
        #val_len = len(v)
        if value in v:
            return k

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    else:
        pass

def read_json(path, source_name):
    with open(os.path.join(path, source_name), 'r', encoding='utf-8') as sourcefile:
        source = json.load(sourcefile)
    return source

def write_txt(path, source_name, context_name, context_len, context):
    outfile = open(os.path.join(path, source_name), 'a', encoding='utf-8')
    outfile.write(context_name + '\n')
    outfile.write(context_len + '\n')
    for i in context:
        note = str(i) + "\n"
        outfile.write(note)
    outfile.close()
#-----------------------------------main-----------------------------
def Val():

    Pred_path = '/disk1/JasonSung/CenterNet1/src/lib/wider_eval/eval_wd/Pred'
    Source_path = '/disk1/JasonSung/CenterNet1/data/widerface/annotations'
    Results_path = '/disk1/JasonSung/CenterNet1/exp/ctdet/coco_dla'
    Fold_path = '/disk1/JasonSung/CenterNet1/src/lib/trains'
    mkdir(Pred_path)

    fold_dict = read_json(Fold_path, 'Fold.json')
    results_dict = read_json(Results_path, 'results.json')

    results_name = 'results.json'
    source_name = 'wf2coco_val.json'


#with open(os.path.join(out_dir, test_name), 'w', encoding='utf-8') as outfile:
#    outfile.write(json.dumps(fold_dict))

    with open(os.path.join(Results_path, results_name), 'r', encoding='utf-8') as readfile:
        result = json.load(readfile)

    with open(os.path.join(Source_path, source_name), 'r', encoding='utf-8') as sourcefile:
        source = json.load(sourcefile)

# the result is a list and the source is a dict

#    file1 = open(DATA_PATH, 'r')
#    lines = count_lines(file1)
    id_dict = {}

    source_list = source["images"]
    for item in source_list:
        key = item["id"]
        value = item["file_name"]
        id_dict[key] = value

#-------------------------------------------------------------------

#make the pred folders previously
    for key in fold_dict:
        key_path = Pred_path + '/' + key
        mkdir(key_path)

#cluster the results.json by image_id
    for key in id_dict:
        container = []
        for item_dict in results_dict:
            if item_dict["image_id"] == int(key):
                x = str(item_dict["bbox"][0])
                y = str(item_dict["bbox"][1])
                w = str(item_dict["bbox"][2])
                h = str(item_dict["bbox"][3])
                score = str(item_dict["score"])
                unit = x + ' ' + y + ' ' + h + ' ' + w + ' ' + h + ' ' + score
                container.append(unit)
        context_len = str(len(container))
        folder_name = get_key(fold_dict, id_dict[key])
        file2 = id_dict[key]
        file_name = id_dict[key].replace('.jpg', '.txt')
        path = os.path.join(Pred_path, folder_name)
        context_name = folder_name + '/' + file2
        write_txt(path, file_name, context_name, context_len, container)


    # the code below is used to generate the id_dict.json / results.json .....
    '''
    with open(os.path.join(out_dir, test_name), 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(fold_dict))

    with open(os.path.join(out_dir, results_name), 'r', encoding='utf-8') as readfile:
        result = json.load(readfile)

    with open(os.path.join(out_dir, source_name), 'r', encoding='utf-8') as sourcefile:
        source = json.load(sourcefile)

    # the result is a list and the source is a dict
    source_list = source["images"]
    for item in source_list:
        key = item["id"]
        value = item["file_name"]
        id_dict[key] = value

    with open(os.path.join(out_dir, id_name), 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(id_dict))
    #print("--------")
    #print(type(source_list))
    DATA_PATH = 'D:\Code\Deep learning\wider_face_split\wider_face_split\wider_face_val_bbx_gt.txt'     #give the file path(could modify here)
    out_dir = 'D:\Code\Deep learning\Test'
    test_name = 'test.json'
    results_name = 'results.json'
    source_name = 'wf2coco_val.json'
    id_name = 'id_dict.json'
    Pred_path = 'D:\Code\Deep learning\Test\pred'

    file1 = open(DATA_PATH, 'r')
    lines = count_lines(file1)

    fold_dict = defaultdict(list)
    id_dict = {}

    for i in range(lines):
        line = linecache.getline(DATA_PATH, i)
        if re.search('jpg', line):
            position = line.index('/')
            file_name = line[position+1:-1]
            folder_name = line[:position]
            fold_dict[folder_name].append(file_name)

        fold_dict = defaultdict(list)
        for i in range(lines):
            line = linecache.getline(DATA_PATH, i)
            if re.search('jpg', line):
                position = line.index('/')
                file_name = line[position+1:-1]
                folder_name = line[:position]
                fold_dict[folder_name].append(file_name)
    '''
