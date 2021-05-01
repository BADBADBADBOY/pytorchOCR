#-*- coding:utf-8 _*-
# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import

import logging

class TrainLog(object):
    def __init__(self,LOG_FILE):
        file_handler = logging.FileHandler(LOG_FILE) #输出到文件
        console_handler = logging.StreamHandler()  #输出到控制台
        file_handler.setLevel('INFO')     #error以上才输出到文件
        console_handler.setLevel('INFO')   #info以上才输出到控制台

        fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        file_handler.setFormatter(formatter) #设置输出内容的格式
        console_handler.setFormatter(formatter)

        logger = logging.getLogger('TrainLog')
        logger.setLevel('INFO')     #设置了这个才会把debug以上的输出到控制台

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        self.logger = logger
        
    def error(self,char):
        self.logger.error(char)
    def debug(self,char):
        self.logger.debug(char)
    def info(self,char):
        self.logger.info(char)
        
class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def set_split(self, names):
        if self.resume:
            pass
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()



