import os.path
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Convert frames into a video')
parser.add_argument('--frames', default='/home/hteam/Documents/hao/Research/Detection/detectron2/result_images', type=str, help='video frames file')
parser.add_argument('--output', default='video', type=str, help='output video name')

def main():
    args = parser.parse_args()
    frames_name= os.listdir(args.frames)
    frames_name.sort()  
    fps = 15
    out = 0
    count = 0
    for frame_name in frames_name:
        print('Processing', frame_name, '...')
        path = os.path.join(args.frames, frame_name)
        frame = cv2.imread(path)
        h, w, _ = frame.shape
        size = (w, h)
        if count == 0:
            out = cv2.VideoWriter(args.output + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            count += 1
        out.write(frame)
    out.release()

if __name__ == '__main__':
    main()