import argparse
import yaml
import subprocess
import os
import sys


def parse_opt():
    parser = argparse.ArgumentParser()

    default_params_path = os.path.join('config', 'training-params.yaml')
    with open(default_params_path, 'r') as file:
        default_params = yaml.safe_load(file)

    parser.add_argument('--img-size', type=int, default=default_params['img_size'],
                        help='image size')
    parser.add_argument('--batch-size', type=int, default=default_params['batch_size'],
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=default_params['epochs'],
                        help='number of epochs')
    parser.add_argument('--data', type=str, default=default_params['data'],
                        help='data configuration path')
    parser.add_argument('--weights', type=str, default=default_params['weights'],
                        help='initial weights path')
    parser.add_argument('--hyp', type=str, default=default_params['hyp'],
                        help='hyperparameters path')
    parser.add_argument('--project', type=str, default=default_params['project'],
                        help='save results to project/name')
    parser.add_argument('--name', type=str, default=default_params['name'],
                        help='save results to project/name')
    parser.add_argument('--cache', action='store_true', help='use cache for images',
                        default=default_params['cache'])

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    command = [sys.executable, 'yolov5/train.py',
               '--img', str(opt.img_size),
               '--batch', str(opt.batch_size),
               '--epochs', str(opt.epochs),
               '--data', opt.data,
               '--weights', opt.weights,
               '--hyp', opt.hyp,
               '--project', opt.project,
               '--name', opt.name]
    if opt.cache:
        command.append('--cache')

    subprocess.run(command)
