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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

parser = argparse.ArgumentParser(description='ACID_Object_Detection_Train')
parser.add_argument('--dataset', default='ACID_dataset', type=str, help='name of dataset')
parser.add_argument('--file', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID_Images', type=str, help='data file')
parser.add_argument('--label', default='/home/hteam/Documents/hao/Research/Dataset/ACID/ACID.json', type=str, help='COCO format json')
parser.add_argument('--model', default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', type=str, help='model')
parser.add_argument('--num_class', default=3, type=int, help='num of classes')
parser.add_argument('--iter', default=300, type=int, help='max iter')

def visualizeDataset(dataset_dicts, ACID_meta):
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=ACID_meta, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    args = parser.parse_args()

    ### regist datasethasattr
    register_coco_instances(args.dataset, {}, args.label, args.file)  
    # detectron2.data.datasets.load_coco_json(args.label, args.file, "ACID_dataset")  # this will set thing_classes = ["excavator", "dump_truck", "cement_truck"]

    ## for keypoint training
    # MetadataCatalog.get("ACID_dataset").keypoint_names = ['body_end_x', 'body_end_y', 'body_end_v', 'cab_boom_x', 'cab_boom_y', 'cab_boom_v', 'boom_arm_x', 'boom_arm_y', 'boom_arm_v', 
    #     'arm_bucket_x', 'arm_bucket_y', 'arm_bucket_v', 'bucket_end_left_x', 'bucket_end_left_y', 'bucket_end_left_v', 'bucket_end_right_x', 'bucket_end_right_y', 'bucket_end_right_v']  
    # MetadataCatalog.get("ACID_dataset").keypoint_flip_map = []
    # MetadataCatalog.get("ACID_dataset").keypoint_connection_rules = []
    ## end for keypoint training

    ### set metadata
    ACID_meta = MetadataCatalog.get(args.dataset)
    dataset_dicts = DatasetCatalog.get(args.dataset)

    ### verify the data loading is correct
    # visualizeDataset(dataset_dicts, ACID_meta)
    
    ### train model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))  

    cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02  #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 128   # default: 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class  # excavator, dump_truck, cement_truck

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    # torch.cuda.empty_cache()
    trainer.train()
    # test(cfg)

if __name__ == '__main__':
    main()