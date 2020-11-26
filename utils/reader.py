import numpy as np


def reader_example(id):
    data_dict = {}
    for i in range(3):
        data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
    data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
    return data_dict


def reader(id):
    return reader_example(id)
