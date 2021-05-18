'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import logging

def Init_logging():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)


def load_records(path):
    records = []
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        a = line.strip().split()
        if a[0]==str(0):
            records.append([])
        else:
            records.append([int(x) for x in a[1:]])
    return records