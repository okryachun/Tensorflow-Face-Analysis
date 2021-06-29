# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:40:33 2021

@author: Oleg Kryachun
"""
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

# local imports
from data_visualization import plot_model_measure, pie_plot
import process_data
import build_model
from run_model import get_model

def train_model(data_path='face_data', evaluate=False, **args):
    """Build and train a new model, then save it.
    
    Parameters
    ----------
        data_path (str): path to directory containing all images
        evaluate (bool): evaluate model after training. Default = False
        **args: extra parameters from argparse
    """
    model_vars = process_data.get_model_vars()
    width, height = model_vars['im_width'], model_vars['im_height']
    race_count , gender_count = model_vars['race_count'], model_vars['gender_count']
    dataset_dict = model_vars['dataset_dict']
    
    # create image attribute dataframe
    img_df = process_data.build_df(data_path, dataset_dict)
    
    # explore data statistics
    plot_data_stats(img_df)
    
    # shuffle dataframe in place
    train_df, test_df = train_test_split(img_df, test_size=0.01)
    
    # create FaceDataGenerator object
    generator = process_data.FaceDataGenerator(img_df, width, height)
    
    # get train, valid, test index splits
    train_idx, valid_idx, test_idx = generator.split_indexes(model_vars['train_split'])
    
    # create CNN Face Analysis Model
    model = build_model.ImageModel().assemble_model(width, height,
                                                    race_count, gender_count)

    # compile model and create custom optimizer
    init_lr = 1e-4
    epochs = 100
    opt = Adam(init_lr, decay=init_lr/epochs)
    
    loss_weights = {"age_out": 4, "race_out": 1.5, "gender_out": 0.1}
    metrics = {"age_out": "mae", "race_out": "accuracy", "gender_out": "accuracy"}
    losses = {"age_out": "mse", "race_out": "categorical_crossentropy",
              "gender_out": "binary_crossentropy"}
    
    model.compile(opt, losses, metrics, loss_weights)

    # initialize hyperparameters for the model
    batch_size = 32
    steps_per_epoch = len(train_idx)//batch_size
    validation_steps = len(valid_idx)//batch_size
    
    # create training and validation data generator objects
    train_gen = generator.generate_images(train_idx, True, race_count,
                                          gender_count, batch_size)
    valid_gen = generator.generate_images(valid_idx, True, race_count, 
                                          gender_count, batch_size)

    # save model checkpoints
    callbacks = [ModelCheckpoint("model/callback_checkpoint/model.hdf5",
                                 monitor="val_loss")]
    
    # fit model
    history = model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=validation_steps)
    
    # overwrites previously saved models
    save_model(model)
    
    # evaluate model if user requested it
    if evaluate:
        evaluate_model(model, history, generator, test_idx, img_df)



def evaluate_model(model, history, generator, test_idx, img_df):
    """Display various model accuracy statistics

    Parameters
    ----------
        model (Model): trained CNN model
        history (History): object containing information about model training
        generator (gen): produces batches of data for model use
        test_idx (list): indices for training data      
        img_df (pd.Dataframe): Dataframe of picture attribute information
    """

    model_vars = process_data.get_model_vars()

    # plot model outcomes
    plot_model_measure(1, history.history['gender_out_accuracy'],
                       history.history['val_gender_out_accuracy'],
                       "Accuracy For Gender Feature", "Accuracy", True)
    
    plot_model_measure(2, history.history['race_out_accuracy'],
                       history.history['val_race_out_accuracy'],
                       "Accuracy For Race Feature", "Accuracy", True)
    
    plot_model_measure(3, history.history['age_out_mae'],
                       history.history['val_age_out_mae'],
                       "Mean Absolute Error for Age Feature", "Accuracy", True)
    
    plot_model_measure(4, history.history['loss'],
                       history.history['val_loss'],
                       "Overall Loss", "Loss", True)
    
    model = get_model()
    # Test trained model on test set
    test_batch_size = 32
    steps = len(test_idx) // test_batch_size
    test_gen = generator.generate_images(test_idx, False, 
                                         model_vars['race_count'],
                                         model_vars['gender_count'],
                                         test_batch_size)
    
    age_pred, race_pred, gender_pred = model.predict(test_gen, steps=steps)
    
    # get number of images return from generator 
    num_batches = int(test_idx.shape[0] / test_batch_size)
    test_idx_len = num_batches * test_batch_size
   
    # get attributes for test data
    images, age_true, race_true, gender_true = get_true_values(img_df, test_idx,
                                                               test_idx_len)
    
    # scale and format attributes back to normal
    race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)
    age_pred = np.round(age_pred * model_vars['max_age'])
   
    # display classification report
    dataset_dict = model_vars['dataset_dict']
    cr_race = cr(race_true, race_pred,
                 target_names=dataset_dict['race_alias'].keys())
    cr_gen = cr(gender_true, gender_pred,
                target_names=dataset_dict['gender_alias'].keys())
    rsq = r2_score(age_true, age_pred)
    
    print("Classification report: Race\n", cr_race)
    print("Classification report:Gender\n", cr_gen)
    
    print('R2 score for age ', rsq)
    
    print("Displaying image prediction matrix")
    
    
    # display image prediction matrix
    image_pred_matrix(images, age_pred, age_true, gender_pred, gender_true,
                      race_pred, race_true, dataset_dict, True)


# Print image predictions
def image_pred_matrix(images, age_pred, age_true, gender_pred, gender_true,
                      race_pred, race_true, dataset_dict, save_img=False):
    """Display image matrix (4x4) showing predicted and true values for images.
    
    Parameters
    ----------
        images (list): pixel image matrices
        age_pred (list): age predictions
        age_true (list): true age values
        gender_pred (list): gender predictions
        gender_true (list): true gender values
        race_pred (list): race predictions
        race_true (list): true race values
        dataset_dict (dict): data to string mapper for img attributes
        save_img (bool): save prediction matrix to local directory
    """
    random_indices = np.random.randint(low=0, high=99, size=16)
    n_cols = 4
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15,17))
    
    for i, img_idx in enumerate(random_indices):
        ax = axes.flat[i]
        ax.imshow(images[img_idx])
                
        ax.set_xlabel('a: {}, g: {}, r: {}'.format(int(age_pred[img_idx]),
                                dataset_dict['gender_id'][gender_pred[img_idx]],
                                   dataset_dict['race_id'][race_pred[img_idx]]))
        
        ax.set_title('a: {}, g: {}, r: {}'.format(int(age_true[img_idx]),
                                dataset_dict['gender_id'][gender_true[img_idx]],
                                   dataset_dict['race_id'][race_true[img_idx]]))
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout(pad=5.0)
    
    if save_img:
        filename = os.path.join("graphs", "pred_matrix")
        plt.savefig(filename)


def get_true_values(df, test_idx, test_idx_len):
    """Retrieve image attributes for the test set

    Parameters
    ----------
        df (pd.Dataframe): image attribute dataframe
        test_idx (list): list of image indeces for test set
        test_idx_len (int): length of test_idx to match generated test set
    Returns
    -------
        images (list): list of test images
        age_true (list): scaled age values for test set
        race_true (list): encoded race list for test set
        gender_true (list): encoded gender list for test set.
    """
    test_idx = test_idx[:test_idx_len]
    count = 0
    images, age_true, race_true, gender_true = [], [], [], []
    for idx in test_idx:
        row = df.iloc[idx]
        # only save 100 images, don't waste too much memory
        if count < 100:
            image = cv2.imread(row.filename)
            images.append(image)
            count += 1
        
        age_true.extend([row.age])
        race_true.extend([row.race_id])
        gender_true.extend([row.gender_id])
        
    age_true = np.array(age_true, dtype='int64')
    race_true = np.array(race_true, dtype='int64')
    gender_true = np.array(gender_true, dtype='int64')
        
    return images, age_true, race_true, gender_true


def save_model(model):
    """Save model to local dir via JSON exporting

    Parameters
    ----------
        model (Model): Trained CNN face model
    """
    if not os.path.isdir("model"):
        os.mkdir("model")
    
    model.save_weights("model/saved_model/model.h5")
    model_json = model.to_json()
    with open("model/saved_model/model.json", "w") as f:
        f.write(model_json)


def plot_data_stats(img_df):
    pie_plot(img_df['gender'], "Gender Distribution")
    pie_plot(img_df['race'], "Race Distribution")
    
    # Split age values into categorical bins for pie plot
    bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
    names = ['<10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
    age_binned = pd.cut(img_df['age'], bins, labels=names)
    
    pie_plot(age_binned, "Age Distribution")
    





















