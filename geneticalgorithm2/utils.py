

#
# 1.5 times faster than row min(a, b)
#
import os


def min_fast(a, b):
    return  a if a < b else b

def max_fast(a, b):
    return a if a > b else b


def folder_create(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)