"""
For classification of images of newspaper pages into
Non Front Page : model outputs 0
Front Page : model outputs 1

Backbone model: ResNeSt https://github.com/zhanghang1989/ResNeSt
Code developed on top of the framework published by cycleGAN https://github.com/junyanz/CycleGAN

Written by the KBR Data Science Lab https://www.kbr.be/en/projects/data-science-lab/
Published at https://github.com/DSL-KBR/newspaper-front-page-recognition
"""

import os
import time
import argparse
from flask import Flask, render_template, send_from_directory

import torch
from models.FPR_ResNeSt import FPRNet
from data.base_dataset import is_image_file
from utils.utils import read_image

import pandas as pd

# set computation device
gpu_ids = [0]
# directory that contains pretrained models
model_dir = r'./checkpoints/training'
# name and training seed of a pretrained model
model_name = r'FPR_ResNeSt'
model_seed = 1
# epoch of a pretrained model
model_epoch = 10

# directory that contains newspaper images
img_dir = r'./FPR testing local'
# directory that contains png images for webpage display
image_dir_web = r'./templates/static'

# folder record: collects all images
images_in_folder = pd.DataFrame(columns=["file name", "front page", "confidence", "path"])

# flask app for serving a webpage
webapp = Flask(__name__)


def fpr_create_model(model_d, model_n, model_s, model_e, gpu):
    model = FPRNet(checkpoints_dir=model_d, model_name=os.path.join(model_n, str(model_s)), gpu_ids=gpu)
    model.isTrain = False
    model.setup(epoch=model_e)
    model.eval()
    return model


def fpr_inference(model, path, path_web):
    """
    1. read images by following {path}
    2. all images will be normalized, resized and combined into a (batch, 3, 1024, 1024) tensor {image_samples}
    3. for each input image, a web version (.png) will be created and be deposited to path_web, the web version
    image will be served by flask later
    4. forward passing {image_samples} through a pretrained FPR model {model}
    5. prediction (front page - 1, non-front page - 0) and confidences are returned
    """
    image_samples = read_image(path, path_web)
    model.set_input(image_samples)
    #    infer_start = time.time()
    model.test()
    #    infer_time = (time.time() - infer_start) / image_samples.shape[0]
    # retrieve prediction from the model
    model_outputs = torch.nn.functional.softmax(model.pred, dim=1).cpu()
    model_outputs = torch.topk(model_outputs, k=1)
    return model_outputs[1], model_outputs[0]


def fpr_broadcast(folder_files, folder_record, new_image_name, new_image_path, new_recognition, new_confidence):
    """
    1. register the newly processed images in {folder_record} (i.e. images_in_folder)
        file name: name of the image file, e.g. BE-KBR00_17172097_19230304_00_01_00_1_01_0001_20562072.tiff
        front page: yes (if inference output is 1), no (if inference output is 0)
        confidnece: a number between 0 and 1 indicating the confidnece on the inference
        path: path to the web version of the image
    2. collect front pages and non-front pages by scanning through the updated {folder_record}
    3. remove record in {folder_record} if the corresponding image is no longer seen in the folder
    """
    for i in torch.arange(start=0, step=1, end=len(new_image_name)):
        folder_record = folder_record.append(pd.DataFrame({'file name': new_image_name[i],
                                                           'front page': 'yes' if new_recognition[i] == 1 else 'no',
                                                           'confidence': new_confidence[i],
                                                           'path': new_image_path[i]}),
                                             ignore_index=True)
    image_front_page = []
    image_non_front_page = []
    file_remove = []
    for index, row in folder_record.iterrows():
        if row['file name'] in folder_files:
            if row['front page'] == 'yes':
                image_front_page.append(os.path.splitext(row['file name'])[0] + '.png')
            else:
                image_non_front_page.append(os.path.splitext(row['file name'])[0] + '.png')
        else:
            file_remove.append(index)

    if len(file_remove) > 0:
        folder_record.drop(file_remove, axis=0, inplace=True)

    return image_front_page, image_non_front_page


@webapp.route('/')
def webpage():
    """
    webpage: main route
    1. check all files in img_dir
    2. register new images with temporary variables:
        image_name: the name of the image files that are not yet registered in {images_in_folder} (new images)
        image_path: path to the new images
        image_path: path to web (.png) version of the new images
    3. pass newly registered images to a trained FPR model for front page recognition
    4. broadcasting recognition results on a webpage using flask
    """
    files = os.listdir(img_dir)

    image_name = []
    image_path = []
    image_path_web = []
    for file in files:
        if is_image_file(file) and not (file in images_in_folder['file name'].values):
            image_name.append(file)
            image_path.append(os.path.join(img_dir, file))
            image_path_web.append(os.path.join(image_dir_web, os.path.splitext(file)[0] + '.png'))

    recognition, confidence = fpr_inference(model=fpr_model, path=image_path, path_web=image_path_web)
    front_page_images, non_front_page_images = \
        fpr_broadcast(files, images_in_folder, image_name, image_path_web, recognition, confidence)

    return render_template('webpage.html', front_pages=front_page_images, non_front_pages=non_front_page_images)


@webapp.route('/templates/static/<filename>')
def serve_image(filename):
    return send_from_directory(image_dir_web, filename)


# crate FPR model
fpr_model = fpr_create_model(model_d=model_dir, model_n=model_name, model_s=model_seed, model_e=model_epoch, gpu=gpu_ids)

if __name__ == '__main__':
    webapp.run(debug=True)
