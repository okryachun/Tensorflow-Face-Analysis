# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:25:14 2021

@author: Oleg Kryachun
"""
from skimage.exposure import adjust_gamma, adjust_sigmoid
from tensorflow.keras.utils import to_categorical
from skimage.util import random_noise
import pandas as pd
import numpy as np
import random
import glob
import cv2
import os

class ImgAttributeError(Exception):
    """Exception mostly for code readability."""
    pass


def build_df(path, dataset_dict, ext="jpg"):
    """Build a Dataframe of attributes for a given `path` of files.
    
        1. Traverse files in `path` and send filename to filename_parser()
        2. Build a Dataframe of attributes for each file in `path`
    Parameters:
    ----------
        path (str): relative or absolute path of image location
        dataset_dict (dict): mapping of attribute values to strings
            ex: "gender": {"male": 1, "female": 0}
        ext (str): image extension type. Default "jpg"
    Returns:
    ----------
        pd.Dataframe: attributes extracted from file names
                [age, gender, gender_id, race, race_id, file]
    """
    def filename_parser(img_path):
        """Split `img_path` into a list of attributes.
        
        Parameters:
        ----------
            img_name (str): full path of image file
        Returns:
        ----------
            list: six attributes extracted from `img_path`
                [age, gender, gender_id, race, race_id, filename]
        Raises:
        ----------
            ImageAttributeError: rasied when `img_path` is incorrectly
                formatted. build_df() skips this image file
        """
        try:
            filename = os.path.split(img_path)[-1:][0]
            age, gender_id, race_id, _ = filename.split('_')
            
            # map gender & race to string values
            gender = dataset_dict['gender_id'][int(gender_id)]
            race = dataset_dict['race_id'][int(race_id)]
    
            return [int(age), gender, gender_id, race, race_id, img_path]
    
        except ValueError:
            # Raised when img_path is incorrectly formatted
            raise ImgAttributeError
    
    image_files = glob.glob(os.path.join(path, "*.%s" % ext))
    attribute_list = []
    
    for img_path in image_files:
        try:
            file_attributes = filename_parser(img_path)
            attribute_list.append(file_attributes)
        except ImgAttributeError:
            print(f"Filename <{img_path}> is incorrectly formatted.\n"
                  "This image will be ignored.")
            continue

    df = pd.DataFrame(attribute_list, columns=["age", "gender", "gender_id",
                                               "race", "race_id", "filename"])
    return df



class FaceDataGenerator():
    """Image data generator/processor for training/testing model."""
    
    def __init__(self, df, im_width, im_height):
        """        
        Parameters
        ----------
            df (pd.Dataframe): dataframe of attributes retrieved from `build_df()`
            im_width (int): image width
            im_height (int): image height
        """
        self.df = df
        self.im_width = im_width
        self.im_height = im_height
        
    def split_indexes(self, train_split):
        """Split df into a 3 lists of indices: train_idx, valid_idx, test_idx.
        
        Parameters:
        ----------
            train_split (int): train ratio
        Return:
        ----------
            3 lists: train_idx, valid_idx, test_idx
        """
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * train_split)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        # further split train_idx by TRAIN_SPLIT to create valid_idx 
        train_up_to = int(train_up_to * train_split)
        valid_idx = train_idx[train_up_to:]
        train_idx = train_idx[:train_up_to]

        # max age to normalize "age" column
        self.max_age = self.df["age"].max()

        return train_idx, valid_idx, test_idx
        
    def preprocess_image(self, img_path, augment):
        """Open, resize, and normalize pixels of an image.
        
        Parameters:
        ----------
            img_path (str): path to image for processing
            augment (bool): decides if image should be augmented
        Return:
        ----------
            np.array: resized & normalized pixel matrix of `img_path`
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        try:
            if augment:
                img = self.augment_img(img)
            img = cv2.resize(img, (self.im_width, self.im_height))
            img = np.array(img) / 255.0
            return img
        except:
            return None

    def augment_img(self, img):
        """Randomly augment images to better represent webcam image quality.

        Parameters
        ----------
            img (np.array): image pixel matrix
        Returns
        -------
            img (np.array): image pixel matrix
        """ 
        n = random.randint(0,4)
        if n == 0:
            img = random_noise(img, var=0.002)
        elif n == 1:
            img = cv2.GaussianBlur(img, (9,9), 0)
        elif n == 2:
            img = np.flipud(img)
        elif n == 3:
            img = adjust_gamma(img, 2)
        elif n == 4:
            img = adjust_sigmoid(img)

        return img
            

    def generate_images(self, img_idx, is_training,
                        race_count, gender_count, batch_size=16):
        """Generate batches of images to feed into CNN model.

        Build a `batch_size` of data and yield the block of data. The data
        consists of images and attributes to each image

        Parameters:
        ----------
            img_idx (list): list of indices
            is_training (bool): determines if in training mode
            race_count (int): race categories
            gender_count (int): gender categories
            batch_size (int): batches to generate. Default = 16
        Yield:
        ----------
            x (np.array): image pixel matrix (preprocessed via preprocess_image())
            y (list): each element in the list is an np.array of attributes.
        """
        # lists to store batched data
        images, ages, races, genders = [], [], [], []
        augment = False
        while True:
            for idx in img_idx:
                person = self.df.iloc[idx]

                age = person["age"]
                race = person["race_id"]
                gender = person["gender_id"]
                file_path = person["filename"]

                if augment and is_training:
                    img = self.preprocess_image(file_path, True)
                    ages.append(age / self.max_age)
                    races.append(to_categorical(race, race_count))
                    genders.append(to_categorical(gender, gender_count))
                    images.append(img)
                    augment = False
                else:
                    augment = True
                    
                img = self.preprocess_image(file_path, False)
                
                ages.append(age / self.max_age)
                races.append(to_categorical(race, race_count))
                genders.append(to_categorical(gender, gender_count))
                images.append(img)

                # yielding condition (when batch_size reached)
                if len(images) >= batch_size:
                    x = np.array(images)
                    y = [np.array(ages), np.array(races), np.array(genders)]
                    yield x, y
                    
                    images, ages, races, genders = [], [], [], []

            if not is_training:
                break


def get_model_vars():
    """Define variables used across this program.
    
    Return
    ------
        model_vars (dict): dict containing all variables and mapping dicts
    """
    
    # dictionary to map image attributes to string values
    dataset_dict = {
        'race_id': {
            0: 'white',
            1: 'black',
            2: 'asian',
            3: 'indian',
            4: 'others'
        },
        'gender_id': {
            0: 'male',
            1: 'female'
        }
    }
    
    # reverse id to alias dicts
    dataset_dict['gender_alias'] = dict((g, i) for i,
                                        g in dataset_dict['gender_id'].items())
    dataset_dict['race_alias'] = dict((r, i) for i,
                                      r in dataset_dict['race_id'].items())
    
    model_vars = {
        'train_split': 0.7,
        'im_width': 256,
        'im_height': 256,
        'gender_count': 2,
        'race_count': 5,
        'max_age': 116,
        'dataset_dict': dataset_dict
    }
    
    return model_vars