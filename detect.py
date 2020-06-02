import argparse
import time
import os
import numpy as np
import json
import cv2
import random

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

parser = argparse.ArgumentParser(description='Multiple_Objects_Detection')
parser.add_argument('--num_class', default=5, type=int, help='num of classes')
parser.add_argument('--frames', default='/home/hteam/Documents/han/paper/data/tracking_mp42frame/test_video_3', type=str, help='video frames file')
parser.add_argument('--cfg', default='./configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', type=str, help='config')
parser.add_argument('--weight', default='/home/hteam/Documents/han/paper/detectron2/output/han/backup/model_final_0218_10000.pth', type=str, help='model weight')
parser.add_argument('--dataset_path', default='/home/hteam/Documents/han/paper/data', type=str, help='dataset path')
parser.add_argument('--label_path', default='/home/hteam/Documents/han/paper/data/annotations.json', type=str, help='label path')
parser.add_argument('--dataset_name', default='wheelchair_dataset', type=str, help='dataset name')
parser.add_argument('--detection_name', default='detection.txt', type=str, help='detection name')


def SetDetector(args):
    ### regist training dataset
    trainingDataset = args.dataset_name
    trainingData_path = args.dataset_path
    label_path = args.label_path
    register_coco_instances(trainingDataset, {}, label_path, trainingData_path) 
    meta = MetadataCatalog.get(trainingDataset)
    DatasetCatalog.get(trainingDataset)

    ### set configure
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.weight)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = (trainingDataset, )
    return cfg, meta

def ReadVideoFrame(path, frame_name):
    frame_id = frame_name[:len(frame_name)-4]
    frame_path = os.path.join(path, frame_name)
    frame = cv2.imread(frame_path)
    return frame_id, frame

def Detect(cfg, meta, image, nameSave='test'):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1],
                metadata=meta, 
                scale=0.8,
                instance_mode=ColorMode.SEGMENTATION 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('./result_images/' + nameSave + '.png', v.get_image()[:, :, ::-1])
    return outputs

def main():
    args = parser.parse_args()
    cfg, meta = SetDetector(args)
    frames_name= os.listdir(args.frames)
    frames_name.sort()
    with open(args.detection_name, 'w') as f:
        count = 0
        for frame_name in frames_name:
            print('Processing', frame_name, '...')
            frame_id, frame = ReadVideoFrame(args.frames, frame_name)
            detection = Detect(cfg, meta, frame, frame_id) 
            detection_field = detection["instances"].get_fields()
            detection_bbox = detection_field['pred_boxes'].tensor.cpu().numpy()
            detection_score = detection_field['scores'].cpu().numpy()
            detection_class = detection_field['pred_classes'].cpu().numpy()
            for i in range(len(detection_class)):
                f.write(frame_id + ',' + str(detection_class[i]) + ',' + str(detection_bbox[i][0]) + ',' + str(detection_bbox[i][1]) + ',' + str(detection_bbox[i][2]) + ',' + str(detection_bbox[i][3]) + ',' + str(detection_score[i]) + '\n')
            count += 1
            
            # if count > 500:
            #     break
        f.close()
    print('Succeed to export detection results to detection.txt')

if __name__ == '__main__':
    main()