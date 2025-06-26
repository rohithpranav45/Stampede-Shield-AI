import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, method='yolo', target='face', yolo_model_path=None, cascade_path='haarcascade_frontalface_default.xml', conf=0.1):
        self.method = method
        self.conf = conf
        self.target = target
        if method == 'yolo':
            if yolo_model_path is None:
                yolo_model_path = 'yolov8n-face.pt' if target == 'face' else 'yolov8x.pt'
            self.model = YOLO(yolo_model_path)
        elif method == 'haar':
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        else:
            raise ValueError('Unknown face detection method')

    def detect(self, frame):
        if self.method == 'yolo':
            # Multi-scale: 1.0x and 2.0x
            boxes = []
            for scale in [1.0, 2.0]:
                img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                results = self.model(img, conf=self.conf)
                if self.target == 'face':
                    b = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.empty((0, 4))
                elif self.target == 'person':
                    b = []
                    for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                        if int(cls) == 0:  # COCO class 0 = person
                            b.append(box)
                    b = np.array(b)
                else:
                    raise ValueError('Unknown target for YOLO')
                if scale != 1.0 and len(b) > 0:
                    b = b / scale  # rescale boxes back to original
                if len(b) > 0:
                    boxes.append(b)
            boxes = np.vstack(boxes) if boxes else np.empty((0, 4))
            return self.nms(boxes, iou_threshold=0.5)
        elif self.method == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self.model.detectMultiScale(gray)
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            return boxes

    def nms(self, boxes, iou_threshold=0.5):
        # Simple NMS for overlapping boxes
        if len(boxes) == 0:
            return boxes
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return boxes[keep] 