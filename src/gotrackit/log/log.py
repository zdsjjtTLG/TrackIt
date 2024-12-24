# -- coding: utf-8 --
# @Time    : 2024/12/23 8:17
# @Author  : TangKai
# @Team    : ZheChengData

import os
import sys
import logging

class LogRecord(object):
    def __init__(self, file_name: str = None, out_fldr: str = None, save_log: bool = True):
        self.file_name = file_name
        # logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        console_handler.setLevel(logging.INFO)

        if save_log:
            if os.path.exists(os.path.join(out_fldr, rf'log')):
                pass
            else:
                os.makedirs(os.path.join(out_fldr, rf'log'))
            file_handler = logging.FileHandler(os.path.join(out_fldr, rf'log', file_name + '.log'), mode='a')
            file_handler.setFormatter(
                logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
            file_handler.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                                handlers=[file_handler, console_handler])
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                                handlers=[console_handler])
        logging.info(rf'{file_name}_logging_info:.....')

    def out_log(self, w: str = None):
        logging.info(self.file_name + ':' + w)
