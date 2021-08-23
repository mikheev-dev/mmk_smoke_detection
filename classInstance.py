from classDraw import DrawFrame
from classReq import MyRequest
from threading import Thread
import datetime


class Instance(MyRequest, DrawFrame):
    def __init__(self,
                 real_number,
                 side,
                 count_border,
                 global_count_border,
                 address,
                 port,
                 x_max, x_min, y_max, y_min,
                 fake_x_max, fake_x_min, fake_y_max, fake_y_min):
        super().__init__(address=address, port=port)
        self.real_number = real_number
        self.side = side
        self.count = 0
        self.global_count = 0
        self.global_count_border = global_count_border
        self.count_border = count_border
        self.timer = 0
        self.last_time = None
        self.imagesSent = 2

        # Coordinates
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

        self.fake_x_max = fake_x_max
        self.fake_x_min = fake_x_min
        self.fake_y_max = fake_y_max
        self.fake_y_min = fake_y_min

    def __increment_count(self):
        """
        Для работы с счетчиками
        Returns: None
        """
        self.count += 1
        if self.count > self.count_border:
            self.global_count += 1
            self.count = 0

    def check_time(self):
        """
        Для работы со временем. Берет время последнего сигнала и сравнивает с нынешним.
        Если разница более минуты, то все счетчики обнуляются
        Returns: minutes (int)
        """
        timer_check = datetime.datetime.now() - self.last_time
        timer_check = timer_check.total_seconds()
        minutes = divmod(timer_check, 60)[0]
        return minutes

    def zeros_counter(self):
        """
        Обнуляет все счетчики
        Returns: None
        """
        if self.timer >= 1:
            self.count = 0
            self.global_count = 0
            self.timer = 0

    def zeros_global(self):
        """
        Обнуляет глобальный счетчик при отправке
        Returns: None
        """
        self.global_count = 0

    def plus_counter(self):
        self.timer = 0
        self.last_time = datetime.datetime.now()
        self.__increment_count()

    def request(self, frame, predict, zone_name):
        if self.global_count == self.global_count_border:
            """
            Отправка сигнала
            """
            if self.imagesSent == 2:
                self.imagesSent = 0
                Thread(target=self.req, args=[self.real_number, self.side, self.fake_x_max,
                                              self.fake_y_max, self.fake_x_min, self.fake_y_min, predict, frame, zone_name]).start()
            else:
                self.imagesSent += 1
                Thread(target=self.req, args=[self.real_number, self.side, self.fake_x_max,
                                              self.fake_y_max, self.fake_x_min, self.fake_y_min, predict, None, zone_name]).start()
            self.zeros_global()

    def checker(self):
        """
        Основная функция, обеспечивающая обновления состояния для каждого инстанса
        Returns: None
        """
        if self.count > 0 or self.global_count > 0:
            self.timer = self.check_time()
            self.zeros_counter()
