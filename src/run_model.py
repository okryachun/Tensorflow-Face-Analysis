# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:35:13 2021

@author: Oleg Kryachun
"""
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
import tarfile
import cv2
import sys
import os

# local import
from process_data import get_model_vars


class FaceNotFound(Exception):
    """Thrown when a face is not found on camera"""
    pass


def crop_face(img):
    """Detect a face and return a cropped image singling out a face.
    
    Parameters
    ----------
        img (np.array): images numpy matrix
    Returns
    -------
        face (np.array): cropped image of a detected face
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')    
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)
        face = np.array(0)
        # if face found
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # extend the size of the face detected
            ext = int(abs(h-y) * 0.5)
            
            # test if extension fits on image, if not ext maximum amount
            if (y+h+ext) > img.shape[0]:
                ext = img.shape[0] - h
            face = img[y:y + h + ext, x:x + w]
            
    # if problem with extracting face, print error and raise FaceNotFound
    except Exception as e:
        print("Error1: ", e)
        raise FaceNotFound
    
    return face


def process_image(img, model_vars):
    """Resize, normalize, and expand dimensions of `img`

    Parameters
    ----------
        img (np.array): image pixel matrix 
        model_vars (dict): various variable values defined for the model
    Returns
    -------
        img (np.array): altered image pixel matrix
    """
    width = model_vars["im_width"]
    height = model_vars["im_width"]
    img = crop_face(img)
    img = cv2.resize(img, (width, height))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def process_results(age, race, gender, model_vars):
    """Select argmax from predictions, map predcitions to string values

    Parameters
    ----------
        age (float): predicted age (normalized)
        race (list): predicted race categorical output list
        gender (list): predicted gender categorical output list
        max_age (int): maximum age that normalized age data
        model_vars (dict): various variable values defined for the model
    Returns
    -------
        age (str): predicted age (unnormalized)
        race (str): predicted race mapped to string value
        gender (str): predicted gender mapped to string value
    """
    max_age = model_vars["max_age"]
    age = str(round(age[0][0] * max_age))
    gender = model_vars["dataset_dict"]['gender_id'][gender.argmax()]
    race = model_vars["dataset_dict"]['race_id'][race.argmax()]
    return age, race, gender


def start_video(model, model_vars):
    """Open video, analyze face using the `model`
    
    Parameters
    ----------
        model (Model): trained CNN model to analyze a face detect on camera
        model_vars (dict): various variable values defined for the model
    """
    vid = cv2.VideoCapture(0)
    counter = 0
    text = ""
    frame_title = "Press q to quit"
    while True:
        # Capture video
        _, frame = vid.read()
        
        # send image to CNN model every 50 iterations
        if counter == 50:
            try:
                img = process_image(frame, model_vars)
            # Error processing image, attempt next frame
            except:
                counter = 49
                continue
  
            age, race, gender = model.predict(img)
            age, race, gender = process_results(age, race, gender, model_vars)
            text = f"Age: {age}, Race: {race}, Gender: {gender}"
            print('Prediction: ', text)
            counter = 0
        
        try:
            # display the resulting frame
            cv2.putText(**optimize_text(text, frame))
            cv2.imshow(frame_title, frame)
        except:
            counter = 49
            continue
        
        # check if q pressed to quit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
        
    vid.release()
    cv2.destroyAllWindows()


def get_model(path):
    """Load pretrained CNN model from local directory
    
    Parameters
    -------
        path (str): path to saved json model
    Returns
    -------
        model (Model): pretrained loaded tensorflow model
    """
    tf.keras.backend.clear_session()

    model_structure_path = os.path.join(path, "model.json")
    model_weights_path = os.path.join(path, "weights.h5")

    if not os.path.exists(model_weights_path):
        tar_weights_path = model_weights_path[:-2] + "tar.gz"
        if os.path.exists(tar_weights_path):
            print(f"Extracting model weights from {tar_weights_path}.")
            tar_data = tarfile.open(tar_weights_path)
            tar_data.extractall(path)
            tar_data.close()
        else:
            print("Error: Missing model weights .h5 file.")
            sys.exit(1)            

    try:
        with open(model_structure_path, 'r') as f:
            loaded_json_model = f.read()

        model = model_from_json(loaded_json_model)
        model.load_weights(model_weights_path)

    except:
        e = sys.exc_info()[0]
        print("Error: ", e)
        print(f"Couldn't load model. Check that {model_weights_path} and {model_structure_path} exist")
        sys.exit(1)
    
    return model


def analyze_picture(model, model_vars, img_path):
    """Run CNN model on an individual image

    Parameters
    ----------
        model (Model): trained CNN model to analyze a face detect in picture
        img_path (str): image path
        model_vars (dict): various variable values defined for the model
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"The path for image: '{img_path}' can't be loaded correctly.")
        return
    
    try:
        img_proc = process_image(img, model_vars)
    except FaceNotFound:
        print("Couldn't find face in image.")
        return
    except Exception as e:
        print(e)
        print("There was a problem processing this image")
        return

    age, race, gender = model.predict(img_proc)
    age, race, gender = process_results(age, race, gender, model_vars)

    # shrink image if too large
    im_shape = list(img.shape)
    if any(i > 900 for i in im_shape):
        scaler = 900/im_shape[0]
        im_shape = (int(im_shape[1]*scaler) , int(im_shape[0]*scaler))
        img = cv2.resize(img, im_shape)

    text = f"Age: {age}, Race: {race}, Gender: {gender}"
    put_text = optimize_text(text, img)
    
    # Display text on frame using cv2.putText() method
    cv2.putText(**put_text)
    cv2.imshow("Face Analyis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_model(model_path, **args):
    """Load model, start live video or individual picture analysis via model

    Parameters
    ----------
        model_path (str): name of model folder in 'saved_models'
        **args : arbitrary parameters from argsparse
    """
    if args['model_type'] == 'normal':
       model_path = 'saved_models/normal_model'

    print(f"Retrieving {args['model_type']} model...")
    model = get_model(model_path)
    print("Model retrieved.")
    model_vars = get_model_vars()
    # start video analysis using model
    if args.get('video', False):
        print("starting video")
        start_video(model, model_vars)
    # if not video, then individual image will be analyzed
    else:
        img_path = args['img_path'][0]
        analyze_picture(model, model_vars, img_path)


def optimize_text(text, img):
    """Adjust text font, size, location for image frame

    Parameters
    ----------
        text (str): string to be displayed on a cv2 frame
        img (np.array): img that will be displayed on frame

    Returns
    -------
        put_text (dict): optimized parameter arguments for cv2.putText
    """
    img_width, img_height, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = cv2.getFontScaleFromHeight(font, round(img_height*0.033))
    thickness = round(scale * 1.8)

    text_size = {"text": text, "fontFace": font,
                 "fontScale": scale, "thickness": thickness}

    org = cv2.getTextSize(**text_size)[0]
    
    # set padding around text
    org = (int(img_width*0.02), int(org[1]*1.3))    

    put_text = {
        "img": img,
        "text": text,
        "fontFace": font,
        "org": org,
        "fontScale": scale,
        "color": (255, 0, 0),
        "thickness": thickness,
        "lineType": cv2.LINE_AA,
        "bottomLeftOrigin": False
    }

    return put_text