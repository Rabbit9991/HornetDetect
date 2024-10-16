import argparse
import cv2
import torch

def detect_video(weights, source, img_size=640, conf_thres=0.5):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # Load video
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        results = model(frame)

        # Process detections
        for pred in results.pred:
            # Filter by confidence threshold
            pred = pred[pred[:, 4] > conf_thres]

            # Draw bounding boxes
            for det in pred:
                xmin, ymin, xmax, ymax, conf, cls = det
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('YOLOv5', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='C:/Users/TaeSan/Desktop/S_P/yolov5-master/runs/train/Test_v1_bat16_100/weights/best.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    opt = parser.parse_args()

    detect_video(opt.weights, opt.source, opt.img_size, opt.conf_thres)
