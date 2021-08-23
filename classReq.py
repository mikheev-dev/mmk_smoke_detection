from classDraw import DrawFrame
import json
import cv2
import base64
import requests
import logging


class MyRequest(DrawFrame):
    def __init__(self, address, port):
        super().__init__(color=(28, 31, 255), thickness=2)
        self.address = address
        self.port = port
        self.M = 1280
        self.N = 720

    def req(self, num, side, x_max, x_min, y_max, y_min, predict, img=None, zone_name=""):
        """
        Отправка сигнала на сервер
        Args:
            num: номер печи
            side: сторона
            x_max: координата первой точки
            x_min: координата второй точки
            y_max: координата первой точки
            y_min: координата второй точки
            predict: процент для отрисовки на изображении
            img: изображения (np.array)
            zone_name: имя зоны
        Returns: None
        """
        data = self.create_json(num, side, x_max, x_min, y_max, y_min, predict, image=img, zone_name=zone_name)
        address_to_server = f"http://{self.address}:{self.port}"
        headers = {'Content-type': 'application/json'}
        request = requests.post(address_to_server, data=data, headers=headers)
        print(request.status_code)

    def create_json(self, number, side, x_max, y_max, x_min, y_min, predict, image=None, zone_name=""):
        """
        Создает необходимый формат для отправки на сервер
        """
        if image is None:
            data = {"data": {"name": side, "number": number, "zone_name": zone_name}}
            raw_data = json.dumps(data)
            return raw_data
        else:
            image = self.draw_rectangle(image, x_max, x_min, y_max, y_min, predict)
            image = self.encode_image(image)
            data = {"data": {"name": side, "number": number, "image": image, "zone_name": zone_name}}
            raw_data = json.dumps(data)
            return raw_data

    def encode_image(self, img):
        """
        Кодирует изображения в строку
        """
        img = cv2.resize(img, (self.M, self.N))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img = cv2.imencode('.jpg', img, encode_param)
        if result:
            frame = base64.b64encode(img).decode("utf-8")
            return frame
        else:
            return False
