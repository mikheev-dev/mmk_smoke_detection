import sys
sys.path.insert(1, "/home/odyssey/mmk_smoke_detection/")

import os
from tensorflow import keras
import tensorflow as tf
import shutil
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_BINARY_MEDIUM_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_splitted_medium_aug_dataset/"
PATH_TO_MEDIUM_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/splitted_medium_aug_dataset/"

PATH_TO_MERGED_BINARY_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_splitted_small_aug_dataset_2_2/"
PATH_TO_MERGED_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/merged_small_aug_dataset_2/"
PATH_TO_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/splitted_small_aug_dataset/"
PATH_TO_BINARY_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_splitted_small_aug_dataset/"

PATH_TO_THREE_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/three_small_aug_dataset/"

PATH_TO_VALIDATION_DATASET = "/home/odyssey/mmk_smoke_detection/validation/expanded_val/"
PATH_TO_THREE_VAL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/three_validation/train"
PATH_TO_BINARY_VAL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_validation/train/"

PATH_TO_HANDLED_THREE_VAL_DATASET = "/home/odyssey/mmk_smoke_detection/quality/handled_validation/"
PATH_TO_HANDLED_BINARY_VAL_DATASET = "/home/odyssey/mmk_smoke_detection/quality/binary_handled_validation/"

# LABELS = ['background', 'emission', 'fire', 'machine']
LABELS = ['background', 'emission', 'fire']
N, M = 224, 224

LOG_DIR = "./mlogs"


from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import math

from typing import Optional

def flow_from_directory(
        path_to_dataset: str,
        is_binary: bool
):
    keras_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=path_to_dataset,
        labels='inferred',
        label_mode='categorical',
        class_names=LABELS[:2] if is_binary else LABELS,
        batch_size=32,
        image_size=(N, M)
    )
    keras_ds = keras_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return keras_ds

def gen_metrics(
        is_binary: bool
) -> list:
    cbs = [keras.metrics.CategoricalAccuracy(),]
    labels = LABELS[:2] if is_binary else LABELS
    for idx, label in enumerate(labels):
        cbs.append(keras.metrics.Precision(
            class_id=idx,
            name=f'precision_{label}'
        ))
        cbs.append(keras.metrics.Recall(
            class_id=idx,
            name=f'recall_{label}'
        ))
    return cbs

def freeze_layers(
        base_model: Model,
        freeze_degree: float
):
    layers_num = len(base_model.layers)
    layers_border = int(freeze_degree * layers_num)
    print("Layers border:", layers_border)
    for idx in range(layers_border):
        base_model.layers[idx].trainable = False

def prepare_model(
        base_model: Model,
        freeze_degree: float,
        is_binary: bool = False,
) -> Model:
    freeze_layers(base_model, freeze_degree)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    if is_binary:
        # TODO попробовать сменить на log
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        predictions = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    #     opt = keras.optimizers.Adam(learning_rate=1e-03)
    opt = keras.optimizers.SGD(learning_rate=1e-02)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy' if is_binary else 'categorical_crossentropy',
        metrics=gen_metrics(is_binary)
    )
    return model

def choose_dataset(is_binary: bool, is_medium: bool) -> str:
    # PATH_TO_BINARY_MEDIUM_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_splitted_medium_aug_dataset/"
    # PATH_TO_MEDIUM_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/splitted_medium_aug_dataset/"
    # PATH_TO_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/splitted_small_aug_dataset/"
    # PATH_TO_BINARY_SMALL_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/binary_splitted_small_aug_dataset/"
    if is_binary:
        #         return PATH_TO_BINARY_MEDIUM_DATASET if is_medium else PATH_TO_BINARY_SMALL_DATASET
        return PATH_TO_BINARY_MEDIUM_DATASET if is_medium else PATH_TO_MERGED_BINARY_SMALL_DATASET
    else:
        return PATH_TO_MEDIUM_DATASET  if is_medium else PATH_TO_MERGED_SMALL_DATASET

def choose_val_dataset(is_binary: bool) -> str:
    #     return PATH_TO_BINARY_VAL_DATASET if is_binary else PATH_TO_THREE_VAL_DATASET
    return PATH_TO_HANDLED_BINARY_VAL_DATASET if is_binary else PATH_TO_HANDLED_THREE_VAL_DATASET

def step_decay(epoch):
    initial_lrate = 5e-3
    drop = 0.6
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def fit_model(
        base_model: Model,
        log_path: str,
        epochs_num: int,
        is_medium: bool = False,
        is_binary: bool = False,
        path_to_dataset: Optional[str] = None,
        path_to_val_dataset: Optional[str] = None,
        freeze_degree: float = 1.0
):
    path_to_dataset = path_to_dataset or choose_dataset(is_binary, is_medium)
    path_to_val_dataset = path_to_val_dataset or choose_val_dataset(is_binary)
    train_set = flow_from_directory(path_to_dataset=os.path.join(path_to_dataset, 'train'), is_binary=is_binary)
    test_set = flow_from_directory(path_to_dataset=os.path.join(path_to_dataset, 'test'), is_binary=is_binary)
    val_set = flow_from_directory(path_to_dataset=path_to_val_dataset, is_binary=is_binary)
    model = prepare_model(base_model=base_model, is_binary=is_binary, freeze_degree=freeze_degree)
    history = model.fit(
        train_set,
        epochs=epochs_num,
        callbacks=[
            keras.callbacks.TensorBoard(log_path),
            keras.callbacks.LearningRateScheduler(step_decay)
        ],
        validation_data=val_set
    )
    print("TEST EPOCH RESULT", model.evaluate(test_set, return_dict=True))
    val_result = model.evaluate(
        val_set,
        return_dict=True
    )
    print("VAL EPOCH RESULT", val_result)
    return val_result


MY_PATH_TO_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/splitted_line_small_aug_dataset"
LINE_VAL_THREE_DATASET = "/home/odyssey/mmk_smoke_detection/augmentation/line_val"


# def save_val_metrics(metrics: list, res_file_name: str, model_name: str, epoch_num: int):
#     class_metrics = metrics[2:]
#     res = []
#     for i in range(0, len(class_metrics), 2):
#         cls_num = i // 2
#         prec, rec = class_metrics[i], class_metrics[i+1]
#         res.append(f"prec{cls_num}: {prec}, rec{cls_num}: {rec}")
#     res = ";\n".join(res)
#     with open(res_file_name, 'a') as r:
#         r.write(f'------- {model_name} ----- {epoch_num} \n')
#         r.write(res)
#         r.write('\n')


RESULT_FILE = 'result.txt'
models = [
    #     ('resnet50', tf.keras.applications.resnet.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))),
    #     ('resnet50', tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    #     ('resnetv2_101_2', tf.keras.applications.resnet_v2.ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    #     ('efficient_net_B6', tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    #     ('densenet', tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
    ('mobilenet', tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
]


if __name__ == "__main__":
    for (model_name, base_model) in models:
        print(f"{model_name}_imagenet_line_three_small_08_logs")
        epoch_num = 50
        val_metrics = fit_model(
            base_model=base_model,
            log_path=f'./{model_name}_imagenet_line_three_small_08_logs',
            epochs_num=epoch_num,
            freeze_degree=0.8,
            path_to_dataset=MY_PATH_TO_DATASET,
            path_to_val_dataset=LINE_VAL_THREE_DATASET
        )
