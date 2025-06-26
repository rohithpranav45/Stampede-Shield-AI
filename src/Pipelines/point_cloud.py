import numpy as np
import cv2

class PointCloudExtractor:
    def __init__(self, homography=None):
        self.homography = homography

    def extract_points(self, boxes, depth_map, image_shape=None):
        points_3d = []
        h_map, w_map = depth_map.shape
        h_img, w_img = image_shape[:2] if image_shape is not None else (h_map, w_map)
        for idx, (x1, y1, x2, y2) in enumerate(boxes):

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # Map (cx, cy) from image to depth map coordinates
            cx_map = int(cx * w_map / w_img)
            cy_map = int(cy * h_map / h_img)
            # Clamp to valid range
            cx_map = np.clip(cx_map, 0, w_map - 1)
            cy_map = np.clip(cy_map, 0, h_map - 1)
            z = float(depth_map[cy_map, cx_map])
            if self.homography is not None:
                uv1 = np.array([[[cx, cy]]], dtype=np.float32)
                xy1 = cv2.perspectiveTransform(uv1, self.homography)[0][0]
                x, y = float(xy1[0]), float(xy1[1])
            else:
                x, y = float(cx), float(cy)
            points_3d.append({"id": idx, "x": x, "y": y, "z": z})
        return points_3d 