import argparse
import time
import os
import numpy as np
import json
import cv2
import random
import torch
from ACID_test import test

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.model_zoo import model_zoo 
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

parser = argparse.ArgumentParser(description='ACID_Object_Detection_Train')
parser.add_argument('--dataset', default='ACID_dataset', type=str, help='name of dataset')
parser.add_argument('--file', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID_train_augmentation', type=str, help='data file')
parser.add_argument('--label', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID_train_augmentation.json', type=str, help='COCO format json')
parser.add_argument('--test_dataset', default='ACID_testing', type=str, help='name of testing dataset')
parser.add_argument('--test_file', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID_testing', type=str, help='testing data file')
parser.add_argument('--test_label', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID_test.json', type=str, help='testing json')
parser.add_argument('--model', default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', type=str, help='model')
parser.add_argument('--weight', default='./output/model_final.pth', type=str, help='model weight')
parser.add_argument('--num_class', default=3, type=int, help='num of classes')
parser.add_argument('--iter', default=30000, type=int, help='max iter')

def main():
    args = parser.parse_args()
    register_coco_instances(args.dataset, {}, args.label, args.file)  # training dataset
    register_coco_instances(args.test_dataset, {}, args.test_label, args.test_file)  # testing dataset

    ### set metadata
    MetadataCatalog.get(args.test_dataset).evaluator_type="coco"
    DatasetCatalog.get(args.test_dataset)

    ### cfg setting
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))  
    cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class  # excavator, dump_truck, cement_truck
    cfg.MODEL.WEIGHTS = args.weight 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = (args.test_dataset,)

    ### trainner setting
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(cfg.MODEL.WEIGHTS)

    ### evaluation setting
    evaluator = COCOEvaluator(args.test_dataset, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, args.test_dataset)
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    main()