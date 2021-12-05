from logging import Logger
from multiprocessing import Process, Queue
from threading import Thread
from typing import List, Callable

import time
import traceback


DEFAULT_SLEEP_TIME = 5


class QLogContext:
    _src_q: Queue
    _dst_q: Queue
    _logger: Logger

    @property
    def logger(self) -> Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        self._logger = logger

    @property
    def src_q(self) -> Queue:
        return self._src_q

    @src_q.setter
    def src_q(self, queue: Queue) -> None:
        self._src_q = queue

    @property
    def dst_q(self) -> Queue:
        return self._dst_q

    @dst_q.setter
    def dst_q(self, queue: Queue) -> None:
        self._dst_q = queue


class BaseProcessTask(QLogContext, Process):
    def __init__(
            self,
            context: QLogContext
    ):
        super().__init__()
        self.src_q = context.src_q
        self.dst_q = context.dst_q
        self.logger = context.logger

    def _setup(self):
        """
            Default wait while inference will up.
        """
        time.sleep(15)

    def _main(self):
        raise NotImplementedError

    def run(self):
        self.logger.info(f"Start process {self.__class__.__name__}")
        self._setup()

        while True:
            try:
                self._main()
            except Exception as e:
                self.logger.exception(traceback.format_exc())


class TaskController:
    _tasks: List[BaseProcessTask]
    _logger: Logger

    def __init__(
            self,
            tasks: List[BaseProcessTask],
            logger: Logger
    ):
        self._logger = logger
        self._tasks = tasks
        for task in tasks:
            self._logger.info(f"TASK.CLASS = {task.__class__.__name__}")

    def start(self):
        for task in self._tasks:
            task.start()

    def terminate(self):
        for task in self._tasks:
            if task.is_alive():
                task.terminate()

    def join(self) -> None:
        while True:
            for task in self._tasks:
                if not task.is_alive():
                    self._logger.error(f"TaskController:: Task {type(task)}:{task.pid} has stopped!")
                    return
            time.sleep(DEFAULT_SLEEP_TIME)


def measure_time_and_log(msg: str):
    def measure_time_and_log_internal(meth: Callable) -> Callable:
        def _wrapper(obj: BaseProcessTask, *args, **kwargs):
            st = time.time()
            result = meth(obj, *args, **kwargs)
            working_time = time.time() - st
            obj.logger.info(f"{msg}:: Handle 1 frame by {working_time:.2f} seconds")
            return result
        return _wrapper
    return measure_time_and_log_internal
