# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:27:10 2021

@author: Oleg Kryachun
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Dense,
                                     BatchNormalization,
                                     Flatten, Dropout)

class ImageModel():
    """Builds a multi-branched model for multiple outputs."""

    def head_branch(self, inputs):
        """Initial convolution layers to each branch.
        
        Parameters
        ----------
            inputs (Input): Input layer from tensorflow.keras.layer.Input
        Returns
        -------
            x (Conv2D): last layer of the convolution chain
        """
        x = Conv2D(16, (3,3), strides=2, padding="same", activation="relu")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(32, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3,3), strides=2, padding="same", activation="relu")(x)

        return x

    def category_branch(self, inputs, n_categories, name):
        """Categorical tail end of CNN branch for gender & race.
        
        Parameters
        ----------
            inputs (Input): Input layer from tensorflow.keras.layer.Input
            n_categories (int): number of outputs for categerical branch
            name (str): name of branch output
        Returns
        -------
            x (Dense): layer that outputs final prediction of branch
        """

        x = self.head_branch(inputs)

        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(n_categories, activation="sigmoid", name=name)(x)

        return x

    def regression_branch(self, inputs, name):
        """Regression tail end of CNN branch for age
        
        Parameters
        ----------
            inputs (Input): Input layer from tensorflow.keras.layer.Input
            name (str): name of branch output
        Returns
        -------
            x (Dense): layer that outputs final prediction of branch
        """

        x = self.head_branch(inputs)

        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, name=name)(x)

        return x

    def assemble_model(self, width, height, race_count, gender_count):
        """Assemble complete model of all branches for CNN
        
        Parameters
        ----------
            width (int): width of input images
            height (int): height of input images
            race_count (int): categories of race
            gender_count (int): categories of gender
        Returns
        -------
            model (Model): fully assembled model of all layers & branches
        """
        input_shape = (width, height, 3)
        
        inputs = Input(shape=input_shape)
        
        age_branch = self.regression_branch(inputs, "age_out")
        gender_branch = self.category_branch(inputs,
                                             gender_count, "gender_out")
        race_branch = self.category_branch(inputs,
                                           race_count, "race_out")

        model = Model(inputs, outputs=[age_branch, race_branch, gender_branch],
                      name="face_model")

        return model