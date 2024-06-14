# Detect and distinguish different boat types from images/videos using YOLO

This project was developed by:

| First name | Last name |           E-mail            |
|:----------:|:---------:|:---------------------------:|
|   Mihály   |  Kocsis   | kocsismhly at gmail dot com |

## Table of Contents
1. [Setup](#setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Training](#training)
4. [Image Detection](#image-detection)
5. [Video Detection](#video-detection)

## Setup

1. Clone this repository

2. Create a virtual environment
    ```bash
    python -m venv .venv
    ```
3. Activate the virtual environment
   - macOS or Linux:
    ```bash
    source .venv/bin/activate
    ```
   - Windows:
    ```bash
    .venv\Scripts\activate
    ```
4. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```

## Data preprocessing

* This project uses the boat dataset from [Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/boat-types-recognition),
  containing the following image-sets:
  * `cruise ship`
  * `ferry boat`
  * `freight boat`
  * `gondola`
  * `inflatable boat`
  * `kayak`
  * `paper boat`
  * `sailboat`
  * `buoy`
* The dataset is split into training and validation sets using an 80/20 ratio.
  * **Training**: 80% of the images from each category.
  * **Validation**: 20% of the images from each category.
* Every image in the dataset has been annotated with bounding boxes around the boats, along with the corresponding class labels.
  * The annotations were performed using `LabelImg` to create the bounding boxes and class labels in `YOLO` format.

## Training

Training the YOLO model on our labeled dataset.

* The default training parameters are specified in `config/training-params.yaml`.
* The hyperparameter tuning settings are specified in `config/tuning.yaml`.

### Default training

To train the model using the default parameters, run the following command:
```bash
python scripts/train.py
```

### Custom training

You can also customize the training parameters according to your needs. For example:
```bash
python scripts/train.py --img 640 --batch 32 --epochs 10 --data config/boat.yaml --weights yolov5s.pt --hyp config/tuning.yaml --name boat_exp --cache
```

#### Parameters

* `img`: Size of the input images
* `batch`: Number of images per batch
* `epochs`: Number of training epochs
* `data`: Path to the dataset configuration file
* `weights`: Path to the initial weights file
* `hyp`: Path to the hyperparameter tuning configuration file
* `name`: Name of the experiment (saving directory)
* `cache`: Whether to cache images for faster training
  
> [!NOTE]
> A pre-trained model is included in the repository to save time, as training can be time-consuming.

## Image detection

The model will draw bounding boxes around detected boats and display the image with the predictions.

### Random image detection

Run the script with the `--random` parameter to display random images from the validation folder.
This mode allows you to cycle through images.

* **Navigation**:
  * Press `space` to see the next random image.
  * Press `q` to quit the program.

```bash
python scripts/image-detect.py --random
```

### Specific image detection

Run the script with the `--image` parameter followed by the path to a specific image file.
This mode displays the detection results for the specified image.

* **Navigation**:
  * Press `q` to quit the program.

```bash
python scripts/image-detect.py --image cruise-ship-holidays-cruise-290913.jpg
```

## Video detection

This script processes a video and draws bounding boxes with predictions around detected boats using the trained model.

### Default video detection

To detect boats from a video using the default parameters, run the following command:
```bash
python scripts/video-detect.py
```
This command uses the example video located in `data/videos/boats.mp4`.

* **Navigation**:
  * Press `q` to quit the program.

### Custom video detection

You can also configure the script to use a different video, skip frames, or resize the video. For example:
```bash
python scripts/video-detect.py --video data/videos/boats.mp4 --frameskip 2 --resize 416
```

#### Parameters

* `video`: Path to the video
* `frameskip`: Number of frames to skip (can help speed up the processing)
* `resize`: Width of the resized video frames
