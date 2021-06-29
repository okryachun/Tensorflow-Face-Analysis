# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:03:01 2021

@author: Oleg Kryachun
"""
import argparse
import sys

sys.path.append('src')
# Local imports
from run_model import run_model
from train_model import train_model


if __name__ == "__main__":
    """Command line interface for training and running the face CNN."""   

    description = "Interact with Face Analysis CNN Model."
    
    parser = argparse.ArgumentParser(description=description)
    subparser = parser.add_subparsers()
    
    run_parser = subparser.add_parser("run", help='Run face analysis model.')
    run_parser.add_argument('--model_path', nargs=1, default='model/saved_model',
                            dest='model_path',
                            help="path of saved trained model")
    
    
    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', action='store_true',
                       help="video cam face analysis")
    group.add_argument('--img', nargs=1, dest='img_path',
                       help='path to image for analysis')
    run_parser.set_defaults(function=run_model)
    
    
    train_parse = subparser.add_parser('train', help='Train CNN model')
    train_parse.add_argument('--eval', action='store_true', dest='evaluate', 
                             help='Evaluate model training statistics.')
    train_parse.add_argument('--model_path', nargs='?', default='model/saved_model',
                             help='location to save trained model')
    train_parse.add_argument('--data_path', nargs='?', dest='data_path',
                            default='face_data', 
                            help="Path to data to train model.")
    train_parse.set_defaults(function=train_model)
    
    
    args = parser.parse_args()
    cmdhelper = args.function
    
    sys.exit(cmdhelper(**vars(args)))