<!-- Banner -->
![Banner](https://github.com/okryachun/random/blob/main/3d-face-scan-banner.jpg)

# Tensorflow Face Analysis

<!-- Brief project description -->

A Tensorflow Python application that predicts age, race, and gender of a given face. This project is mostly an exploratory project to better understand convolution neural networks and Tensorflow.

This project showcases two models: 
- the **facenet** model is trained using the FaceNet facial recognition neural network developed by researchers at Google in 2015
- the **normal** model is trained from scratch using my understanding of convolution neural networks (CNN)

# Demo-Preview

An example of both CNN models in action using live video feed from a webcam. 

*(The movement was necessary so that there could be slightly different images fed to the models as they continuously make predictions.)*

***P.S. I am 20 years old, white, and male.***

#### **facenet** CNN video feed demo:

![Facenet GIF](https://github.com/okryachun/random/blob/main/facenet_video.gif)

#### **normal** CNN video feed demo:

![Normal GIF](https://github.com/okryachun/random/blob/main/normal_video.gif)


# Table of contents

- [Project Title](#tensorflow-face-analysis)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Development](#development)
- [Credits](#credits)
- [License](#license)

# Installation
[(Back to top)](#table-of-contents)

#### Download Code:
```git clone https://github.com/okryachun/Tensorflow-Face-Analysis.git```

#### Set up environment:
**Mac with M1 chip** instructions: https://medium.com/codex/installing-tensorflow-on-m1-macs-958767a7a4b3 
- (The M1 chip utilizes ARM64 which is different than Intel chips, therefore there is a special Tensorflow required for this.)

**Linux**, **Mac (without M1 chip)** could set up virtual environments two different ways:
1. Using `conda` instructions: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
    - When creating a `conda` environment, use the `environment.yml` file to include all necessary packages.
    - ex: ```conda env create --file environment.yml --name tf_env```

2. Using `pipenv` instructions: https://pipenv-fork.readthedocs.io/en/latest/basics.html
    - `pipenv` will automatically utilize the `Pipfile` in the project directory 
    - ex: ```pipenv install``` (execute from project directory)

**Windows** is best to run with Spyder creating a `conda` virtual environment:
1. Install Spyder and set up a virtual environment instructions: http://docs.spyder-ide.org/current/installation.html 


# Usage
[(Back to top)](#table-of-contents)

#### Program main functions:
1. Run a trained model (two models to choose from):
    - Two ways to run:
        1. Video feed utilizing a webcam to capture continuous images
            - ```python src/main.py run --video```
        2. Individual image passed as a command line argument for analysis
            - ```python src/main.py run --img path/to/img```
    - Specify a specific model (**facenet** is the default model):
        1. Using the **normal** model
            - ```python src/main.py run --video --model_type normal```
        

2. Train a model (two models to choose from):
    - Train **normal** model:
        - ```python src/main.py train --model_type normal```
    - Display and save post training model evaluation graphs and statistics
        - ```python src/main.py train --eval```

#### Project Directory Outline/Purpose

Directories:
- `model_structure`: contains images of both CNN architectures
- `saved_model`: contains model structure and weights for three separate models
    1. `facenet_model`: transfer learned model aka **facenet** model
    2. `facenet_transfer_layer`: pre-trained model create by Google researchers, this model is used as an internal layer inside the **facenet** model
    3. `normal_model`: pre-trained model structure and weights of CNN trained from scratch
- `src`: contains all the source code for this project
- `statistics`: contains various statistics collected throughout the projects creation
    - `data_stats`: statistics about the UTK face data
        1. `adjusted_data_stats`: post data altering to include more underrepresented age groups
        2. `raw_data_stats`: raw UTK data stats
- `model_metrics`: contains model evaluation data for both trained models **facenet** & **normal**
    - `facenet_model_stats`: **facenet** post model training performance evaluations
    - `normal_model_stats`: **normal** post model training performance evaluations
- `xml`: contains the .xml file to load the opencv face detecting model

Files:
- `dataset_1.tar.gz` & `dataset_2.tar.gz`: contain half of the UTK face data in each zipped file. A single file was too large to upload to Github.
- `environment.yml`: contains necessary packages for creating a `conda` virtual environment
- `Pipfile`: contains necessary packages for creating a `pipenv` virtual environment


# Model Evaluation
[(Back to top)](#table-of-contents)

### Facenet Model
**facenet** transfer learned CNN model using the FaceNet pre trained model created by Google researchers in 2015 for facial recognition.

![Facenet Age](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/facenet_model_stats/Mean%20Absolute%20Error%20for%20Age%20Feature.png)

![Facenet Gender](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/facenet_model_stats/Accuracy%20For%20Gender%20Feature.png)

![Facenet Race](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/facenet_model_stats/Accuracy%20For%20Race%20Feature.png)

---

### Normal Model

**normal** CNN model trained from scratch. 

![Normal Age](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/normal_model_stats/Mean%20Absolute%20Error%20for%20Age%20Feature.png)

![Normal Gender](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/normal_model_stats/Accuracy%20For%20Gender%20Feature.png)

![Normal Race](https://github.com/okryachun/Tensorflow-Face-Analysis/blob/master/statistics/model_metrics/normal_model_stats/Accuracy%20For%20Race%20Feature.png)


# Development
[(Back to top)](#table-of-contents)

Most development would likely be done adjusting various model architectures, the easiest way to do this would be to edit `src/build_model.py`. Otherwise feel free to adjust or change anything about this project.

# Credits
[(Back to top)](#table-of-contents)

- FaceNet facial recognition model: https://github.com/davidsandberg/facenet
- UTK Face Dataset: https://susanqq.github.io/UTKFace/
- This project was inspired by Rodrigo Bressan: https://github.com/rodrigobressan/keras-multi-output-model-utk-face


# License
[(Back to top)](#table-of-contents)

MIT License

Copyright (c) [2021] [Oleg Kryachun]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
