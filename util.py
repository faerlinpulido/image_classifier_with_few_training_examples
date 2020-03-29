#####################################################
# File: util.py
# Author: Faerlin Pulido
# Email: pulido.faerlin@gmail.com
#####################################################
# Utility functions for fetching and preparing 
# the food-101 dataset from 
# http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
#####################################################
# IMPORTANT: These functions assume that the extracted
# dataset reside in the directory data/food-101

import os
import json
import shutil
import imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path

fname = 'food-101.tar.gz'
origin = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
base_dir = 'data'

# Directories of food-101 dataset and meta data
food_dir = path.join(base_dir, 'food-101')
images_dir = path.join(food_dir, 'images')
meta_dir = path.join(food_dir, 'meta')
train_json = path.join(meta_dir, 'train.json')
test_json = path.join(meta_dir, 'test.json')

# 101 Classes: Train, Validation, and Test directories
folder_101 = path.join(base_dir, '101_classes')
full_train_dir = path.join(folder_101, 'full_train')
train_dir = path.join(folder_101, 'train')
valid_dir = path.join(folder_101, 'valid')
test_dir = path.join(folder_101, 'test')

# 10 Classes: Train, Validation, and Test directories
folder_10 = path.join(base_dir, '10_classes')
full_train_10_dir = path.join(folder_10, 'full_train')
train_10_dir = path.join(folder_10, 'train')
valid_10_dir = path.join(folder_10, 'valid')
test_10_dir = path.join(folder_10, 'test')

def download():
    """Download the food-101 dataset in a folder called datasets"""
    
    tf.keras.utils.get_file(
        fname=fname,
        origin=origin,
        extract=True,
        cache_dir='.')
    
def get_classes():
    """Return a list of classes for the food-101 datasets"""
    
    with open(train_json) as file:
        train_dict = json.load(file)
    return list(train_dict.keys())
    
def create_full_train_folder(labels, dst_dir):
    """Copies all images from the food-101/train folder belonging to the given labels.
       to the given destination folder. 
    
    Args:
        labels (list): list of labels (classes)
        dst_dir (str): new folder that will contain the copied images
    """
    
    with open(train_json) as file:
        train_dict = json.load(file)
        copy_images(train_dict, labels, images_dir, dst_dir)
        
def create_train_and_validation_folders(labels, train_dir, valid_dir, validation_split=0.2):
    """Copies images from the food-101/train and split them into training and 
       validation sets belonging to the given labels to the given destination folders.
    
    Args:
        labels (list): list of labels (classes)
        train_dir (str): new folder that will contain the training set
        valid_dir (str): new folder that will contain the validation set
    """
    
    with open(train_json) as file:
        train_dict = json.load(file)
        copy_train_valid(train_dict, labels, images_dir, train_dir, valid_dir, validation_split)

def create_test_folder(labels, dst_dir):
    """Copies images from the food-101/test folder belonging to the given labels. 
       The images are copied to the given destination folder.
    
    Args:
        labels (list): list of labels (classes)
        dst_dir (str): new folder that will contain the copied images.
    """
    
    with open(test_json) as file:
        test_dict = json.load(file)
        copy_images(test_dict, labels, images_dir, dst_dir)

def copy_images(images_dict, labels, src_dir, dst_dir):
    """Copies images from the source directory to the destination directory belonging
       to the given labels. 
    
    Args:
        images_dict (dict): maps labels to the image filenames
        labels (list): list of labels (classes)
        src_dir (str): source of image files
        dst_dir (str): destination of copied image files
    """
    
    images_subset = get_subset(images_dict, labels)
    for label in labels:
        directory = os.path.join(dst_dir, label)
        os.makedirs(directory)

    for label in labels:
        count = 0
        for filename in images_subset[label]:
            src = os.path.join(src_dir, filename + '.jpg')
            dst = os.path.join(dst_dir, filename + '.jpg')
            shutil.copyfile(src, dst) 
            count += 1
        print(label,' ', count)

def copy_train_valid(images_dict, labels, src_dir, train_dir, valid_dir, validation_split=0.2):
    """Copies images from the source folder, split them into training and validations sets, and 
       copies them to the given destination folders.
    """
    
    images_subset = get_subset(images_dict, labels)
        
    images_list = []
    labels_list = []
    for label in labels:
        for image_path in images_subset[label]:
            images_list.append(image_path)
            labels_list.append(label)
    images_list = np.array(images_list)
    labels_list = np.array(labels_list)
        
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
    generator = splitter.split(images_list, labels_list)
    train_index, valid_index = next(generator)
        
    train_images_list = images_list[train_index]
    train_labels_list = labels_list[train_index]
    valid_images_list = images_list[valid_index]
    valid_labels_list = labels_list[valid_index]
        
    train_dict = to_dict(train_images_list, train_labels_list)
    valid_dict = to_dict(valid_images_list, valid_labels_list)
        
    copy_images(train_dict, labels, src_dir, train_dir)
    copy_images(valid_dict, labels, src_dir, valid_dir)

def create_folder_with_single_image(folder, category, filename):
    category_folder = os.path.join(folder, category)
    if (not os.path.exists(category_folder)):
        os.makedirs(category_folder)
    
    src = os.path.join(train_dir, category, filename)
    dst = os.path.join(category_folder, category + '.jpg')
    shutil.copyfile(src, dst)
        
def to_dict(images_list, category_list):
    images = {}
    N = len(images_list)
    for index in range(N):
        image = images_list[index]
        category = category_list[index]
        if category not in images:
            images[category] = []
        images[category].append(image)         
    return images
                
def get_subset(dictionary, keys):    
    subset = {}
    for key in keys:
        subset[key] = dictionary[key]
    return subset 

def plot_images(instances, image_size, images_per_row=10, **options):
    """Plot images of size image_size-by-image_size-by-3

    Args:
        instances (list of numpy arrays): Each array is an image of size image_size-by-image_size-by-3
        images_per_row (int, optional): Number of images per row. Defaults to 10.

    Reference: code is adopted from 
        Geron A: 2017, "Hands-On Machine Learning with Scikit-Learn and Tensorflow". 
    """

    images_per_row = min(len(instances), images_per_row)
    images = [instance for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((image_size, image_size * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.figure(figsize=(12,20))
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")