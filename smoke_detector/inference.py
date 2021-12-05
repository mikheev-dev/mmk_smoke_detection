import numpy as np

from data import QBatch, QFrame
from tasks import BaseProcessTask, measure_time_and_log, QLogContext
from typing import Any


class Inference(BaseProcessTask):
    _model: Any
    _path_to_model: str

    def __init__(
            self,
            context: QLogContext,
            path_to_model: str,
    ):
        super().__init__(context)
        self._path_to_model = path_to_model

    def _setup(self):
        import tensorflow as tf
        import tensorflow.keras as keras

        gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
                ]
            )
        self._model = keras.models.load_model(self._path_to_model, compile=True)
        self.logger.info(f"{self.__class__.__name__}:: Success load model.")

    def _get_batch(self) -> QBatch:
        return self.src_q.get()

    @measure_time_and_log(msg="Inference:: Handle frames in batch")
    def _handle_batch(
            self,
            batch: QBatch
    ) -> np.ndarray:
        return self._model.predict_on_batch(batch.batch)

    def _split_batch(
            self,
            batch_predictions: np.ndarray,
            batch: QBatch
    ):
        batch_offset = 0
        for frame in batch.frames:
            for box in frame.boxes:
                box.emission_prediction = batch_predictions[batch_offset]
                batch_offset += 1

    def _push(
            self,
            batch: QBatch
    ):
        for frame in batch.frames:
            frame.draw_and_save(path_to_save="test_inference")
            self._dst_q.put(frame)
            self.logger.info(f"{self.__class__.__name__}: Put 1 image to q")

    def _main(self):
        batch: QBatch = self._get_batch()
        self.logger.info(f"{self.__class__.__name__}: Get 1 batch {batch.batch.shape} from q")
        batch_predictions = self._handle_batch(batch)
        self._split_batch(
            batch_predictions=batch_predictions,
            batch=batch
        )
        self._push(batch)
