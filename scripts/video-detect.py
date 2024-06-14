import torch
import cv2
import argparse

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/boats/weights/best.pt')

class_names = ['cruise ship', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat', 'kayak', 'paper boat',
               'sailboat', 'buoy']


def draw_boxes_cv2(frame, results):
    for *box, conf, cls in results:
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def detect_boats_in_video(video_path, frame_skip, resize_width):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Error: Could not open video {video_path}')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        print(f'Error: Could not retrieve video properties for {video_path}')
        return

    scale = resize_width / width
    resize_height = int(height * scale)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (resize_width, resize_height))

        results = model(frame)
        predictions = results.xyxy[0].numpy()

        frame_with_boxes = draw_boxes_cv2(frame, predictions)
        cv2.imshow('Detected Boats', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect boats in a video using YOLOv5.')
    parser.add_argument('--video', type=str, default='data/videos/boats.mp4', help='Path to the video file')
    parser.add_argument('--frameskip', type=int, default=1, help='Number of frames to skip')
    parser.add_argument('--resize', type=int, default=640, help='Width to resize the video frames')

    args = parser.parse_args()

    detect_boats_in_video(args.video, frame_skip=args.frameskip, resize_width=args.resize)
    print('Press \'q\' to stop playing the video.')
