from typing import List
from config import Config


class RoiLoader:
    coordinates_roi: List
    fake_coordinates_roi: List
    numbers: List
    side: List

    def __init__(self,
                 jd: Config):
        coord_machine = jd.return_roi_machine()
        coord_coke = jd.return_roi_coke()
        coord_machine_fake = jd.return_fake_machine()
        coord_coke_fake = jd.return_fake_coke()

        self.coordinates_roi = []
        self.fake_coordinates_roi = []
        self.numbers = []
        self.side = []
        # загрузка координат ROI
        if coord_coke is not None and coord_coke_fake is not None:
            start_end_coke = jd.return_number_coke()
            if start_end_coke is None:
                exit("Координаты зон существуют, а номера нет")

            numbers_coke = [x for x in range(start_end_coke[0], start_end_coke[1] + 1)]
            side_coke = start_end_coke[2]
            if len(numbers_coke) != len(coord_coke):
                exit("Количество зон не совпадает с количеством номеров")

            for i in range(len(coord_coke)):
                self.coordinates_roi.append(coord_coke[str(i)])
                self.fake_coordinates_roi.append(coord_coke_fake[str(i)])
                self.numbers.append(numbers_coke[i])
                self.side.append(side_coke)

        if coord_machine is not None and coord_machine_fake is not None:
            start_end_machine = jd.return_number_machine()
            if start_end_machine is None:
                exit("Координаты зон существуют, а номера нет")

            numbers_machine = [x for x in range(start_end_machine[0], start_end_machine[1] + 1)]
            side_machine = start_end_machine[2]
            if len(numbers_machine) != len(coord_machine):
                exit("Количество зон не совпадает с количеством номеров")

            for i in range(len(coord_machine)):
                self.coordinates_roi.append(coord_machine[str(i)])
                self.fake_coordinates_roi.append(coord_machine_fake[str(i)])
                self.numbers.append(numbers_machine[i])
                self.side.append(side_machine)
