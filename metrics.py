import argparse
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Draw Metrics')
parser.add_argument('--metrics', default='/home/hteam/Documents/hao/Research/Detection/detectron2/output/metrics_faster_rcnn_X_101_32x8d_FPN_3x_cfg4.json', type=str, help='metric')
parser.add_argument('--txt', default=None, type=str, help='invalid json format')
parser.add_argument('--name', default='test', type=str, help='output json name')
parser.add_argument('--output', default='./result_images/', type=str, help='output charts')

def txt_to_json(path, name):
    f = open(path, 'r')
    lines = f.readlines()
    json = []
    for i in range(len(lines)):
        if i == 0:
            obj = '[' + lines[i] + ','
        elif i == len(lines) - 1:
            obj = lines[i] + ']'
        else:
            obj = lines[i] + ','
        json.append(obj)
    g = open(name + '.json', 'w')
    for line in json:
        g.write(line)
    return name + '.json'

def DrawMetric(x, y, xlabel, ylabel, title, path):
    print('Processing', title, '...')
    plt.plot(x, y)
    # plt.ylim(0.6, 1)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    path = os.path.join(path, title + '.png')
    plt.savefig(path)
    plt.clf()

def main():
    args = parser.parse_args()
    path = args.metrics

    if args.txt != None:
        name = txt_to_json(args.txt, args.name)
        path = name

    with open(path) as f:
        metrics = json.load(f)
        cls_accuracy = []
        iteration = []
        loss_box_reg = []
        loss_cls = []
        total_loss = []

        for metric in metrics:
            cls_accuracy.append(metric['fast_rcnn/cls_accuracy'])
            iteration.append(metric['iteration'])
            loss_box_reg.append(metric['loss_box_reg'])
            loss_cls.append(metric['loss_cls'])
            total_loss.append(metric['total_loss'])
            
        print('Last Iteration:', iteration[-1])
        print('Classification Accuracy:', cls_accuracy[-1])
        print('Box Regression Loss:', loss_box_reg[-1])
        print('Classification Loss:', loss_cls[-1])
        print('Total Loss:', total_loss[-1])

        # cls_accuracy
        DrawMetric(np.array(iteration), np.array(cls_accuracy), 'Iteration', 'Accuracy', 'Classification Accuracy', args.output)

        # loss_box_reg
        DrawMetric(np.array(iteration), np.array(loss_box_reg), 'Iteration', 'Loss', 'Box Regression Loss', args.output)

        # loss_cls
        DrawMetric(np.array(iteration), np.array(loss_cls), 'Iteration', 'Loss', 'Classification Loss', args.output)

        # total_loss
        DrawMetric(np.array(iteration), np.array(total_loss), 'Iteration', 'Loss', 'Total Loss', args.output)
    f.close()

if __name__ == '__main__':
    main()