import cv2
import argparse


parser = argparse.ArgumentParser(description='Cut MP4 to frame')
parser.add_argument('--video_path', type=str, help='video path')
parser.add_argument('--save_path', type=str, help='the path to save frames')



def main():
    args = parser.parse_args()
    print(cv2.__version__)
    vidcap = cv2.VideoCapture(args.video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(args.save_path + "/frame_%06d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print ('Read a new frame'+ str(count) +': ', success)
        count += 1


if __name__ == '__main__':
    main()