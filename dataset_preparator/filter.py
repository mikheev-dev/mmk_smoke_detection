import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import glob
import os
import random
from cropper_generator import print_labels_stats
from IPython.display import display, clear_output
import shutil
import tqdm
import random

from multiprocessing import Queue, Process

from preparator import DatasetDirectoryController, LABELS

from typing import List, Any, Optional

PATH_TO_DATASET = "filtered_dataset_copy"
PATH_TO_EMISSION = os.path.join(PATH_TO_DATASET, "emission")
PATH_TO_BACKGROUND = os.path.join(PATH_TO_DATASET, "background")

print(matplotlib.get_backend())

fig = plt.figure()


def save_img_pathes_to_file(
        path_to_stored_images: str,
        images: List[str]
):
    if os.path.exists(path_to_stored_images):
        return
    with open(path_to_stored_images, 'w') as st:
        for image_path in images[:-1]:
            st.write(f"{image_path}\n")
        st.write(image_path)
    print(len(images))


PATH_TO_STORED_GLOB = os.path.join(PATH_TO_DATASET, "stored_glob.txt")
PATH_TO_ALREADY_HANDLED = os.path.join(PATH_TO_DATASET, "all_handled.txt")


PATH_TO_READY_DATASET = "ready_dataset_copy"


def extract_with_already_handled(
    dircon: DatasetDirectoryController,
    label_idx: int,
    path_to_dataset: str,
) -> int:
    path_to_already_handled = os.path.join(path_to_dataset, "all_handled.txt")
    path_to_label_files = dircon.get_directory_for_label_idx(label_idx=label_idx)
    label = LABELS[label_idx]
    with open(path_to_already_handled, "r") as a:
        already_handled_files = list(map(lambda x: x[:-1], a.readlines()))

    label_files = glob.glob(os.path.join(path_to_dataset, label, "*.jpg"))
    # print(os.path.join(path_to_dataset, label, "*.jpg"), label_files[0: 5], already_handled_files[0:5])
    dataset_label_files = set(label_files) & set(already_handled_files)
    print(f"LEN {label}: {len(dataset_label_files)}")
    for label_file in tqdm.tqdm(dataset_label_files):
        shutil.copy2(label_file, path_to_label_files)
    return len(dataset_label_files)


# def extract_background(dircon: DatasetDirectoryController) -> int:
#     path_to_dataset_background_files = dircon.get_directory_for_label_idx(label_idx=0)

#     with open(PATH_TO_ALREADY_HANDLED, "r") as a:
#         already_handled_files = list(map(lambda x: x[:-1], a.readlines()))
#     background_files = glob.glob(os.path.join(PATH_TO_BACKGROUND, "*.jpg"))
#     dataset_background_files = set(background_files) & set(already_handled_files)
#     for background_file in tqdm.tqdm(dataset_background_files):
#         shutil.copy2(background_file, path_to_dataset_background_files)
#     return len(path_to_dataset_background_files)


def extract_without_already_handled(
    dircon: DatasetDirectoryController,
    label_idx: int,
    path_to_dataset: str,
    dataset_size: int
):
    ready_label_dir = dircon.get_directory_for_label_idx(label_idx=label_idx)
    path_to_files = glob.glob(os.path.join(path_to_dataset, LABELS[label_idx], "*.jpg"))
    print(f"LEN {LABELS[label_idx]}: {dataset_size}")
    for _ in tqdm.tqdm(range(dataset_size)):
        img_path = random.choice(path_to_files)
        shutil.copy2(img_path, ready_label_dir)


FUTURE_DIR = 'future'


class DatasetFilter:
    _glob_path: str
    _all_handled_path: str
    _label_path: Optional[str]
    _dataset_path: str

    _dir_controller: DatasetDirectoryController
    _glob_images: List[str]

    _already_handled_count: int

    _images: List[str]
    _already_handled_class_count: int

    def __init__(
        self,
        path_to_dataset: str,
        glob_rule: str,
        dir_controller: DatasetDirectoryController,
        label_idx: Optional[int] = None,
    ):
        self._dataset_path = path_to_dataset
        label = LABELS[label_idx] if label_idx else 'all'
        self._label_path = os.path.join(
            path_to_dataset,
            LABELS[label_idx]
        ) if label_idx else None
        self._dir_controller = dir_controller
        self._glob_path = os.path.join(
            path_to_dataset,
            f"{label}_stored_glob.txt"
        )
        self._all_handled_path = os.path.join(
            path_to_dataset,
            f"{label}_all_handled.txt"
        )
        save_img_pathes_to_file(
            path_to_stored_images=self._glob_path,
            images=glob.glob(glob_rule)
        )

        self._load_glob()
        self._load_already_handled()
        self._images = self._glob_images[self._already_handled_count:]
        print(f"Images to handle: {len(self._images)}")

    def _load_glob(self):
        with open(self._glob_path) as g:
            self._glob_images = [x[:-1] for x in g.readlines()]

    def _load_already_handled(self):
        if os.path.exists(self._all_handled_path):
            with open(self._all_handled_path, "r") as a:
                already_handled_files = list(map(lambda x: x[:-1], a.readlines()))
                self._already_handled_class_count = self._count_already_handled_of_current_label(
                    already_handled_files
                ) if self._label_path else len(already_handled_files)
                self._already_handled_count = len(already_handled_files)
        else:
            self._already_handled_count = 0
            self._already_handled_class_count = 0
        print(f"Already handled {self._already_handled_count}")

    def _count_already_handled_of_current_label(
        self,
        already_handled_files: List[str]
    ) -> int:
        label_files = glob.glob(os.path.join(self._label_path, "*.jpg"))
        return len(set(label_files) & set(already_handled_files))

    @staticmethod
    def _plot_img_and_get_label(fig, ax, img_path: str) -> int:
        img = Image.open(img_path)
        ax.imshow(np.asarray(img))
        display(fig)
        label_idx = int(input(f"Input real class for image {img_path}:")) - 1
        clear_output(wait=True)
        return label_idx

    def handle_images(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        with open(self._all_handled_path, "a") as f:
            for img_path in self._images:
                print(f"Already handled of all {self._already_handled_count}/{len(self._glob_images)}")
                print(f"Already handled class count {self._already_handled_class_count}")
                user_label_idx = self._plot_img_and_get_label(fig, ax, img_path)
                f.write(f"{img_path}\n")
                if user_label_idx == 5:
                    self._move_to_future_dir(img_path)
                    continue
                if user_label_idx == 4:
                    self._handle_delete_file(img_path)
                    continue
                self._handle_entered_label_idx(
                    image_path=img_path,
                    label_idx=user_label_idx
                )
                self._already_handled_count += 1

    def _handle_entered_label_idx(self,
                                  image_path: str,
                                  label_idx: int):
        current_label_idx = LABELS.index(os.path.dirname(image_path).split('/')[-1])
        if current_label_idx == label_idx:
            print("Label was correct")
            self._already_handled_class_count += 1
            return
        self._move_img_to_label_dir(
            image_path=image_path,
            label_idx=label_idx
        )

    def _handle_delete_file(self, img_path: str):
        os.remove(img_path)
        print("Removed from dataset")
        
    def _move_to_future_dir(
        self,
        image_path: str
    ):
        future_dir = os.mkdir(os.path.join(self._dataset_path, FUTURE_DIR))
        if not os.path.exists(future_dir):
            os.mkdir(future_dir)
        new_image_path = os.path.join(future_dir, os.path.basename(image_path))
        os.replace(image_path, new_image_path)

    def _move_img_to_label_dir(self,
                               image_path: str,
                               label_idx: int):
        target_dir = self._dir_controller.get_directory_for_label_idx(label_idx)
        new_image_path = os.path.join(target_dir, os.path.basename(image_path))
        os.replace(image_path, new_image_path)
        print(f"Moved to {LABELS[label_idx]} dir")


TO_REMOVE = 'remove'
TO_MOVE = 'move'
TO_STAY = 'stay'
TO_FUTURE = 'future'

class FilterTaskHandler(Process):
    _q: Queue

    def __init__(
        self,
        q: Queue
    ):
        super().__init__()
        self._q = q

    def handle_from_q(
        self
    ):
        action, dst_dir, img_path = self._q.get()
        if action == TO_REMOVE:
            os.remove(img_path)
            print("FilterTaskHandler: Removed from dataset")
        elif action == TO_MOVE:
            label = dst_dir.split('/')[-1]
            new_image_path = os.path.join(dst_dir, os.path.basename(img_path))
            os.replace(img_path, new_image_path)
            print(f"FilterTaskHandler: Moved to {label} dir")
        elif action == TO_FUTURE:
            new_image_path = os.path.join(dst_dir, os.path.basename(img_path))
            os.replace(img_path, new_image_path)
    
    def run(self):
        while True:
            self.handle_from_q()



class MPDatasetFilter(DatasetFilter):
    _q: Queue
    _thandler: FilterTaskHandler    

    def __init__(
        self,
        path_to_dataset: str,
        glob_rule: str,
        dir_controller: DatasetDirectoryController,
        label_idx: Optional[int] = None,
    ):
        super().__init__(
            path_to_dataset=path_to_dataset,
            glob_rule=glob_rule,
            dir_controller=dir_controller,
            label_idx=label_idx,
        )
        self._q = Queue()
        self._thandler = FilterTaskHandler(self._q)

    def _move_img_to_label_dir(self,
                               image_path: str,
                               label_idx: int):
        target_dir = self._dir_controller.get_directory_for_label_idx(label_idx)
        self._q.put(
            (TO_MOVE, target_dir, image_path)
        )

    def _handle_delete_file(self, img_path: str):
        self._q.put(
            (TO_REMOVE, '', img_path)
        )

    def _move_to_future_dir(
        self,
        image_path: str
    ):
        future_dir = os.path.join(self._dataset_path, FUTURE_DIR)
        if not os.path.exists(future_dir):
            os.mkdir(future_dir)
        self._q.put(
            (TO_FUTURE, future_dir, image_path)
        )
        
    def handle_images(self):
        self._thandler.start()
        super().handle_images()


# def MPFilter:
#     _dfilter: MPDatasetFilter
#     _thandler: FilterTaskHandler
#     _q: Queue

#     def __init__(
#         self,
#         dircon: DatasetDirectoryController
#     ):
#         self._q = Queue()
#         self._dfilter = MPDatasetFilter(self._q, dircon)
#         self._thandler = DatasetDirectoryController(self._q)

#     def run(self):
#         self._dfilter.run()


if __name__ == "__main__":
    dircon = DatasetDirectoryController(dataset_main_dir=PATH_TO_READY_DATASET)
    dircon.prepare_directories()
    dataset_label_size = extract_emission(dircon)
    print(dataset_label_size)
    for label_idx in [0, 2, 3]:
        extract_not_emission(label_idx=label_idx,
                             dataset_size=dataset_label_size,
                             dircon=dircon)
