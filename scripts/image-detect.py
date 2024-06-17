import pathlib
import torch
import os
import cv2
import argparse4
import random
from pathlib import Path
import platform

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

model_path = Path('runs', 'train', 'boats', 'weights', 'best.pt').as_posix()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

class_names = ['cruise ship', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat',
               'kayak', 'paper boat', 'sailboat', 'buoy']


def draw_boxes_cv2(image, results):
    for *box, conf, cls in results:
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[int(cls)]} {conf:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image


def detect_boats(image_path):
    image = cv2.imread(image_path)
    results = model(image_path)
    predictions = results.xyxy[0].numpy()
    image_with_boxes = draw_boxes_cv2(image, predictions)
    cv2.imshow('Detected Boats', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_random_images(directory):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    while True:
        image_path = random.choice(image_files)
        image = cv2.imread(image_path)
        results = model(image_path)
        predictions = results.xyxy[0].numpy()
        image_with_boxes = draw_boxes_cv2(image, predictions)
        cv2.imshow('Detected Boats', image_with_boxes)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect boats in images using YOLOv5.')
    parser.add_argument('--image', type=str,
                        help='Path to a specific image file')
    parser.add_argument('--random', action='store_true',
                        help='Display predictions for random images from the validation folder')

    args = parser.parse_args()

    if args.random:
        print('Press \'space\' to see the next image.')
        print('Press \'q\' to quit.')
        val_images_path = os.path.join('data', 'boat-dataset', 'images', 'val')
        detect_random_images(val_images_path)
    elif args.image:
        print('Press \'q\' to quit.')
        detect_boats(args.image)
    else:
        print('Please provide an image path with --image or use --random to display random images.')
