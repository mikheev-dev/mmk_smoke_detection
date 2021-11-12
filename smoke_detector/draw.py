import cv2
import numpy as np


class DrawFrame:
    def __init__(self, color, thickness):
        self.color = color
        self.thickness = thickness

    def draw_rectangle(self,
                       frame, x_max, x_min, y_max, y_min, predict):
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.color, self.thickness)
        proc = np.around(float(predict[np.argmax(predict)]), 2) * 100
        cv2.putText(frame, f'Emissions:{int(proc)}%',
                    (x_min - 5, y_max + 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return frame
