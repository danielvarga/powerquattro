import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

from base import *


if __name__ == "__main__":
    pkl_filename, = sys.argv[1:]

    with open(pkl_filename, "rb") as f:
        date, records = pickle.load(f)

    print(date, len(records))

    secs = np.array([record.SaveSec for record in records])
    m0 = np.array([record.Moduls0 for record in records])
    m1 = np.array([record.Moduls1 for record in records])
    module_counts = np.array([len(record.modules) for record in records])
    print(module_counts.min(), module_counts.max())
    module_count_max = module_counts.max()
    kept_indices = [i for i, record in enumerate(records) if len(record.modules) == module_count_max]
    print(len(records) - len(kept_indices), "records discarded because of missing module data")
    kept_secs = secs[kept_indices]

    key = "PSol"
    for module_index in range(module_count_max):
        kept_data = np.array([getattr(records[i].modules[module_index], key) for i in kept_indices])
        plt.plot(kept_secs, kept_data)
    plt.show()
