
import os


def read_data(path):
    with open(path, encoding='utf-8') as f:
        all_data = f.read().split("\n")
    