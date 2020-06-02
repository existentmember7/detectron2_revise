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

parser = argparse.ArgumentParser(description='ACID_Object_Detection_Test')
parser.add_argument('--dataset', default='ACID_dataset', type=str, help='name of dataset')
parser.add_argument('--file', default='/home/hteam/Documents/han/paper/data/Images_RGB', type=str, help='data file')
parser.add_argument('--label', default='/home/hteam/Documents/han/paper/data/annotations.json', type=str, help='COCO format json')
parser.add_argument('--cfg', default='./configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', type=str, help='config')
parser.add_argument('--weight', default='/home/hteam/Documents/han/paper/detectron2/output/han/backup/model_final_0218_10000.pth', type=str, help='model weight')
parser.add_argument('--image', default=None, type=str, help='image for testing')
parser.add_argument('--name', default='test', type=str, help='name for test image')
parser.add_argument('--num_class', default=5, type=int, help='num of classes')

def test(dataset, weight, cfg, test_image_path=None, test_image_name=None):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weight)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = (dataset, )
    predictor = DefaultPredictor(cfg)

    ACID_meta = MetadataCatalog.get(dataset)
    dataset_dicts = DatasetCatalog.get(dataset)

    if test_image_path == None:
        for d in random.sample(dataset_dicts, 5):    
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)

            v = Visualizer(im[:, :, ::-1],
                        metadata=ACID_meta, 
                        scale=0.8,
                        instance_mode=ColorMode.SEGMENTATION 
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite('./result_images/' + d["file_name"][52:58] + '.png', v.get_image()[:, :, ::-1])
           # cv2.imshow(d["file_name"], v.get_image()[:, :, ::-1])
           # cv2.waitKey(0)
           # cv2.destroyAllWindows()
    else:
        im = cv2.imread(test_image_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=ACID_meta, 
                    scale=0.8,
                    instance_mode=ColorMode.SEGMENTATION 
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('./han_result/result/' + test_image_name + '.png', v.get_image()[:, :, ::-1])
       # cv2.imshow(test_image_path, v.get_image()[:, :, ::-1])
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()

def main():
    args = parser.parse_args()

    register_coco_instances(args.dataset, {}, args.label, args.file) 

    cfg = get_cfg()
    cfg.merge_from_file(args.cfg) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class  # excavator, dump_truck, cement_truck

    test(args.dataset, args.weight, cfg, args.image, args.name)

if __name__ == '__main__':
    main()
