import argparse
import yaml
import subprocess
import os
import sys
import pathlib
import platform
from pathlib import Path

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def parse_opt():
    parser = argparse.ArgumentParser()

    default_params_path = Path(os.path.join('config', 'training-params.yaml')).as_posix()
    with open(default_params_path, 'r') as file:
        default_params = yaml.safe_load(file)

    parser.add_argument('--img-size', type=int, default=default_params['img_size'],
                        help='image size')
    parser.add_argument('--batch-size', type=int, default=default_params['batch_size'],
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=default_params['epochs'],
                        help='number of epochs')
    parser.add_argument('--data', type=str, default=str(Path(default_params['data']).as_posix()),
                        help='data configuration path')
    parser.add_argument('--weights', type=str, default=str(Path(default_params['weights']).as_posix()),
                        help='initial weights path')
    parser.add_argument('--hyp', type=str, default=str(Path(default_params['hyp']).as_posix()),
                        help='hyperparameters path')
    parser.add_argument('--project', type=str, default=str(Path(default_params['project']).as_posix()),
                        help='save results to project/name')
    parser.add_argument('--name', type=str, default=default_params['name'],
                        help='save results to project/name')
    parser.add_argument('--cache', action='store_true', help='use cache for images',
                        default=default_params['cache'])

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    command = [sys.executable, str(Path('yolov5', 'train.py').as_posix()),
               '--img', str(opt.img_size),
               '--batch', str(opt.batch_size),
               '--epochs', str(opt.epochs),
               '--data', str(Path(opt.data).as_posix()),
               '--weights', str(Path(opt.weights).as_posix()),
               '--hyp', str(Path(opt.hyp).as_posix()),
               '--project', str(Path(opt.project).as_posix()),
               '--name', opt.name]
    if opt.cache:
        command.append('--cache')

    subprocess.run(command)
