import json
import os


class JsonData:
    """
    Класс, позвляющий работать с Json файлами
    """
    def __init__(self, name):
        """
        :param name: имя файла (обязательно указывать формат)
        """
        self.name = name
        self.mode = None
        self.start_oven = None
        self.end_oven = None
        self.camera_address = None
        self.model_gassing = None
        self.model_nameplate = None
        self.signal_server_address = None
        self.image_server_address = None
        self.roi_ovens = {}
        self.roi_nameplate = {}

        self.json = None

        filename, file_extension = os.path.splitext(self.name)
        if str(file_extension) != '.json':
            exit('Необходимо указать расширение файла .json')

        if not os.path.exists(self.name):
            self.json = self.__create_json()
        else:
            self.json = self.__load_json()

    @staticmethod
    def __create_json():
        """
        Создает дефолтный Json файл
        :return:
        """
        data = {'camera_address': None,
                'count': 4,
                'global_count': 5,
                'side': 'M',
                'start': 701,
                'end': 765,
                'server_address': '0.0.0.0',
                'server_port': 0000,
                'model_path': 'model/leaks.h5',
                'matrix': None,
                'roi': {
                    'roi': None},
                }
        # with open(f'{self.name}', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        return data

    def save_json(self):
        """
        Сохраняет Json файл, внося в него измнения, если они были
        :return:
        """
        with open(f'{self.name}', 'w') as f:
            f.write(json.dumps(self.json))

    def __load_json(self):
        with open(f'{self.name}', 'r') as f:
            json_data = json.load(f)
            return json_data

    def create_roi(self, roi):
        """
        Создает зоны интереса в Json
        :param roi: координаты
        :return: None
        """
        rois = {idx: {'x_min': r[0][0], 'y_min': r[0][1], 'x_max': r[1][0], 'y_max': r[1][1]}
                for idx, r in enumerate(roi)}
        self.json['roi'] = rois
        self.roi_ovens = rois

    def create_fake_roi(self, roi):
        rois = {idx: {'x_min': r[0][0], 'y_min': r[0][1], 'x_max': r[1][0], 'y_max': r[1][1]}
                for idx, r in enumerate(roi)}
        self.json['fake_roi'] = rois

    def create_matrix(self, matrix):
        self.json['matrix'] = matrix

    def return_rois(self):
        return self.json['roi']

    def return_fake_rois(self):
        return self.json["fake_roi"]

    def return_counter(self):
        return self.json['count'], self.json['global_count']

    def return_address(self):
        return self.json['camera_address']

    def return_side(self):
        return self.json['side']

    def return_start_end(self):
        return self.json['start'], self.json['end']

    def server_data(self):
        return self.json['server_address'], self.json['server_port']

    def return_model(self):
        return self.json['model_path']

    def return_matrix(self):
        return self.json['matrix']

    def return_roi_machine(self):
        return self.json['roi_machine']

    def return_roi_coke(self):
        return self.json['roi_coke']

    def return_number_machine(self):
        return self.json['number_machine']

    def return_number_coke(self):
        return self.json['number_coke']

    def return_fake_machine(self):
        return self.json["fake_machine"]

    def return_fake_coke(self):
        return self.json["fake_coke"]
