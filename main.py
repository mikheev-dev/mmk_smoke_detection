import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from classInstance import Instance
from classJson import JsonData
from threading import Thread
from tensorflow import keras
import tensorflow as tf
import numpy as np
import queue
import time
import sys


def analyse_jetson(my_queue):
    """
    Функция анализа зон интереса
    returns: None
    """

    print('Start analyse module')
    labels = {idx: my_class for idx, my_class in
              enumerate(['background', 'emission', 'fire', 'machine'])}

    cap = cv2.VideoCapture(camera_address)
    ret, frame = cap.read()

    while not ret:
        cap.release()
        cap = cv2.VideoCapture()
        ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        while not ret:
            print('NO CONNECTION')
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_address)
            ret, frame = cap.read()
            if ret:
                print("GOOD")

        clear = frame.copy()

        if data is not None:
            frame = cv2.warpPerspective(frame, Matrix, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

        arr = np.zeros([len(coordinates_roi), N, M, 3])
        try:
            for idx in range(len(coordinates_roi)):
                r_ = coordinates_roi[idx]
                crop_frame = frame[r_['y_min']:r_['y_max'], r_['x_min']:r_['x_max']]
                crop_frame = cv2.resize(crop_frame, (M, N))
                crop_frame = np.reshape(crop_frame, (1, N, M, 3))
                arr[idx] = crop_frame
        except Exception as error:
            print(error)

        arr = preprocess_input(arr)
        predict = model.predict_on_batch(arr)

        my_queue.put([predict, clear])


def logic_jetson(my_queue):
    global zone_name
    """
    Функция для второго потока, которая берет из очереди значения предсказания модели
    Returns:
    """
    print('Start logic module')
    # загрузка количества распознанных газований, необходимых для отправки сигналов
    # и изобржений на сервер
    count, global_count = jd.return_counter()

    instance_bat = [Instance(real_number=numbers[num],
                             side=side[num],
                             count_border=count,
                             global_count_border=global_count,
                             address=server_address,
                             port=server_port,
                             x_max=coordinates_roi[num]['x_max'],
                             x_min=coordinates_roi[num]['x_min'],
                             y_max=coordinates_roi[num]['y_max'],
                             y_min=coordinates_roi[num]['y_min'],
                             fake_x_max=fake_coordinates_roi[num]['x_max'],
                             fake_x_min=fake_coordinates_roi[num]['x_min'],
                             fake_y_max=fake_coordinates_roi[num]['y_max'],
                             fake_y_min=fake_coordinates_roi[num]['y_min'])
                    for num in range(len(coordinates_roi))]

    time_sleep = 0.01
    while True:
        if my_queue.qsize() > 0:
            item = my_queue.get()
            my_queue.task_done()

            for idx, r in enumerate(item[0]):
                if np.argmax(item[0][idx]) == 1 and item[0][idx][1] >= 0.75:
                    # обновление счетчика распознанных газований
                    instance_bat[idx].plus_counter()
                    # отправка сигнала на сервер
                    instance_bat[idx].request(item[1], item[0][idx], zone_name)

        for inst in instance_bat:
            inst.checker()

        time.sleep(time_sleep)


if __name__ == '__main__':

    global_queue = queue.Queue(maxsize=100)
    zone_name = sys.argv[1]
    # загрузка файла конфигурации JSON
    jd = JsonData(f'conf/conf_{zone_name}.json')

    coord_machine = jd.return_roi_machine()
    coord_coke = jd.return_roi_coke()
    coord_machine_fake = jd.return_fake_machine()
    coord_coke_fake = jd.return_fake_coke()

    coordinates_roi = []
    fake_coordinates_roi = []
    numbers = []
    side = []
    # загрузка координат ROI
    if coord_coke is not None and coord_coke_fake is not None:
        start_end_coke = jd.return_number_coke()
        if start_end_coke is not None:
            numbers_coke = [x for x in range(start_end_coke[0], start_end_coke[1] + 1)]
            side_coke = start_end_coke[2]
            if len(numbers_coke) != len(coord_coke):
                exit("Количество зон не совпадает с количеством номеров")
        else:
            exit("Координаты зон существуют, а номера нет")

        for i in range(len(coord_coke)):
            coordinates_roi.append(coord_coke[str(i)])
            fake_coordinates_roi.append(coord_coke_fake[str(i)])
            numbers.append(numbers_coke[i])
            side.append(side_coke)

    if coord_machine is not None and coord_machine_fake is not None:
        start_end_machine = jd.return_number_machine()
        if start_end_machine is not None:
            numbers_machine = [x for x in range(start_end_machine[0], start_end_machine[1] + 1)]
            side_machine = start_end_machine[2]
            if len(numbers_machine) != len(coord_machine):
                exit("Количество зон не совпадает с количеством номеров")
        else:
            exit("Координаты зон существуют, а номера нет")

        for i in range(len(coord_machine)):
            coordinates_roi.append(coord_machine[str(i)])
            fake_coordinates_roi.append(coord_machine_fake[str(i)])
            numbers.append(numbers_machine[i])
            side.append(side_machine)

    server_address, server_port = jd.server_data()
    model_path = jd.return_model()
    N, M = 75, 75  # размер входного изображения
    camera_address = jd.return_address()
    model = keras.models.load_model(model_path, compile=True)

    # загрузка матрицы трансформации перспективы
    data = jd.return_matrix()
    if data is not None:
        Matrix = np.array(data["matrix"])
        maxWidth = data["maxWidth"]
        maxHeight = data["maxHeight"]

    # Start application
    Thread(target=logic_jetson, args=[global_queue]).start()
    analyse_jetson(my_queue=global_queue)
