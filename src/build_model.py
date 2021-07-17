# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:27:10 2021

@author: Oleg Kryachun
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Dense,
                                     BatchNormalization,
                                     Dropout, GlobalAveragePooling2D)

# local imports
from run_model import get_model

class ImageModel():
    """Builds a multi-branched model for multiple outputs."""

    def head_layer(self, inputs):
        """Initial convolution layers to each branch.
        
        Parameters
        ----------
            inputs (Input): Input layer from tensorflow.keras.layer.Input
        Returns
        -------
            x (Conv2D): last layer of the convolution chain
        """
        x = Conv2D(16, (3,3), strides=1, padding="same", activation="relu")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(16, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3,3), strides=1, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3,3), strides=2, padding="same", activation="relu")(x)
        
        return x
           
    def facenet_head(self, inputs):
        """Load facenet model, exlude last few layers after "add_41"

        Parameters
        ----------
            inputs (Input): Input layer from tensorflow.keras.layer.Input
        Returns
        -------
            facenet_model (Model): facenet model with trimmed bottleneck layers
        """
        facenet_model = get_model("saved_models/facenet_transfer_layer")

        # get layer before bottlneck
        facenet_model = Model(facenet_model.input,
                       outputs=facenet_model.get_layer("add_41").output,
                       name='facenet_transfer_layer')

        # Set last 20 layers to trainable
        facenet_model.trainable = False
        for layer in facenet_model.layers[-20:]:
            layer.trainable = True
            
        # convert model to sequential layer
        facenet_model = facenet_model(inputs )
        
        return facenet_model

    def category_branch(self, head_layer, n_categories, name):
        """Categorical tail end of CNN branch for gender & race.
        
        Parameters
        ----------
            facenet_model (Model): trained facenet model
            n_categories (int): number of outputs for categerical branch
            name (str): name of branch output
        Returns
        -------
            x (Dense): layer that outputs final prediction of branch
        """
        x = GlobalAveragePooling2D()(head_layer)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(n_categories, activation="sigmoid", name=name)(x)

        return x

    def regression_branch(self, head_layer, name):
        """Regression tail end of CNN branch for age
        
        Parameters
        ----------
            facenet_model (Model): trained facenet model
            name (str): name of branch output
        Returns
        -------
            x (Dense): layer that outputs final prediction of branch
        """
        x = GlobalAveragePooling2D()(head_layer)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(1, name=name)(x)

        return x

    def assemble_model(self, model_type, input_shape, race_count, gender_count):
        """Assemble complete model of all branches for CNN
        
        Parameters
        ----------
            model_type (str): name of model type to build ('facenet' or 'normal')
            input_shape (list): input shape of image for CNN model
            race_count (int): categories of race
            gender_count (int): categories of gender
        Returns
        -------
            model (Model): fully assembled model of all layers & branches
        """
        if model_type == 'facenet':
            head_layer = self.facenet_head
        elif model_type == 'normal':
            head_layer = self.head_layer

        inputs = Input(shape=input_shape)
    
        head = head_layer(inputs)

        age_branch = self.regression_branch(head, "age_out")
        gender_branch = self.category_branch(head, gender_count, "gender_out")
        race_branch = self.category_branch(head, race_count, "race_out")

        model = Model(inputs, outputs=[age_branch, race_branch, gender_branch],
                      name="face_model")

        return model
