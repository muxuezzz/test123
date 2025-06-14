#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: a simple pipeline system based on multiprocessing
Author: MindX SDK
Create: 2022
History: NA
"""

import logging
import time
from abc import abstractmethod
from multiprocessing import Process, Queue, Manager
from typing import Tuple


class MetaData:
    """
    the metadata to pass some info between workers
    """

    def __init__(self):
        self.profiler = []


class FinishMsg(object):
    """
    the message to notify presenter agent exit
    """
    pass


class PipelineWorker:
    def __init__(self, do_profiling=False):
        self.do_profiling = do_profiling
        self.times_e2e = Manager().list()
        self.times_process = Manager().list()
        self.times_output = Manager().list()
        self.times_input = Manager().list()

    def init(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def loop(self):
        self.init()
        while True:
            if not self.input_q.empty():
                e2e_start = time.time()
                input_start = time.time()
                data = self.input_q.get()
                self.times_input.append(time.time() - input_start)
                process_start = time.time()
                if isinstance(data, FinishMsg):
                    break
                if isinstance(data, Tuple):
                    new_data = self.process(*data)
                else:
                    new_data = self.process(data)
                self.times_e2e.append(time.time() - process_start)
                if isinstance(new_data[-1], MetaData):
                    new_data[-1].profiler.append(time.time() - process_start)
                output_start = time.time()
                self.output_q.put(new_data)
                self.times_output.append(time.time() - output_start)
                self.times_process.append(time.time() - e2e_start)
            else:
                time.sleep(0.001)

        if self.do_profiling:
            logging.info(f"----------------------------------------------")
            logging.info(f"profiling of module: {self.__class__.__name__}")
            try:
                logging.info(f"e2e time:{sum(self.times_process) / len(self.times_process) * 1000} ms")
                logging.info(f"process time:{sum(self.times_e2e) / len(self.times_e2e) * 1000} ms")
                logging.info(f"input time: {sum(self.times_input) / len(self.times_input) * 1000} ms")
                logging.info(f"output time: {sum(self.times_output) / len(self.times_output) * 1000} ms")
            except ZeroDivisionError:
                logging.info(f"no data put in this module")
            logging.info(f"----------------------------------------------")

    def start(self, input_q, output_q):
        self.input_q = input_q
        self.output_q = output_q
        self.agent_process = Process(target=self.loop)
        self.agent_process.start()
        time.sleep(1)
        if not self.agent_process.is_alive():
            logging.error("process init failed!")
            raise ValueError

    def end(self):
        if self.agent_process.is_alive():
            while not self.input_q.empty:
                self.input_q.get()
            self.input_q.put(FinishMsg())
            self.agent_process.join()


class PipelineSystem:
    def __init__(self, q_size, module_list):
        length = len(module_list)
        self.module_list = []
        q_list = [Queue(q_size)]

        for i in range(length):
            q = Queue(q_size)
            try:
                self.module_list.append(module_list[i])
                module_list[i].start(q_list[-1], q)
            except:
                self.end()
                raise ValueError("PipelineSystem init failed.")
            q_list.append(q)
        self.q_list = q_list

    def send(self, x):
        self.q_list[0].put(x)

    def get(self, timeout=10):
        res = self.q_list[-1].get(timeout=timeout)
        return res

    def end(self):
        for module in self.module_list:
            module.end()

    def get_input_queue_state(self):
        return self.q_list[0].qsize()

    def get_output_queue_state(self):
        return self.q_list[-1].qsize()


class MultiPipelineSystem:
    def __init__(self, parallel_num, q_size, module_list):
        self.pipelines = []
        for i in range(parallel_num):
            from copy import deepcopy
            self.pipelines.append(PipelineSystem(q_size, deepcopy(module_list)))
        self.send_cursors = Queue()

    def send(self, x):
        q_states = [p.get_input_queue_state() for p in self.pipelines]
        ind = q_states.index(min(q_states))
        self.pipelines[ind].send(x)
        self.send_cursors.put(ind)

    def get(self, timeout=None):
        ind = self.send_cursors.get()
        res = self.pipelines[ind].get(timeout)
        return res

    def end(self):
        for pipeline in self.pipelines:
            pipeline.end()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)


    class A(PipelineWorker):
        def process(self, x, y):
            print("Here A")
            time.sleep(0.1)
            return x * 2, y * 2


    class B(PipelineWorker):
        def process(self, x, y):
            print("Here B")
            time.sleep(0.1)
            return x * 3, y * 3


    class C(PipelineWorker):
        def process(self, x, y):
            print("Here C")
            time.sleep(0.1)
            return x * 4, y * 4


    parallel_num = 8
    workers = [A(), B(), C()]
    pipeline = MultiPipelineSystem(parallel_num, 5, workers)


    def send(pipeline):
        for data in range(5):
            s = (data, data + 1)
            pipeline.send(s)


    send_p = Process(target=send, args=(pipeline,))
    send_p.start()

    start = time.time()
    for data in range(5):
        res = pipeline.get(10)
        logging.info(res)
    logging.info(f"e2e time :{time.time() - start}")

    send_p.join()
    pipeline.end()
