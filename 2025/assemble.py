import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import collections
from datetime import datetime

from base import *


def extract_data(pkl_filename, expected_module_count, key):
    with open(pkl_filename, "rb") as f:
        date, records = pickle.load(f)

    print(date, len(records))

    secs = np.array([record.SaveSec for record in records])
    m0 = np.array([record.Moduls0 for record in records])
    m1 = np.array([record.Moduls1 for record in records])
    module_counts = np.array([len(record.modules) for record in records])
    kept_indices = [i for i, record in enumerate(records) if len(record.modules) == expected_module_count]
    discarded_indices = [i for i, record in enumerate(records) if len(record.modules) != expected_module_count]

    def print_histogram(numbers):
        counts = collections.Counter(numbers)  # Count occurrences
        for key in sorted(counts):  # Sort keys
            print(f"{key}: {counts[key]}", end=", ")
        print()


    print("from file", pkl_filename, len(discarded_indices), "records discarded because of missing (or superfluous) module data")
    if len(discarded_indices) < 5:
        print("namely these seconds:", secs[discarded_indices])

    print("histogram of module counts:", end=' ')
    print_histogram(module_counts)

    kept_secs = secs[kept_indices]
    if len(kept_secs) == 0:
        return np.zeros((0, 13)), date

    modulewise = []
    for module_index in range(expected_module_count):
        kept_data = np.array([getattr(records[i].modules[module_index], key) for i in kept_indices])
        modulewise.append(kept_data)
    all_data = np.concatenate([kept_secs[:, None]] + modulewise, axis=-1)
    return all_data, date


def main_vis():
    pkl_filename, = sys.argv[1:]
    key = "PSol"

    all_data = extract_data(pkl_filename, expected_module_count=6, key=key)
    kept_secs = all_data[:, 0]
    aggregated = all_data[:, 1:].sum(axis=-1)
    plt.plot(kept_secs, aggregated)
    plt.show()
    exit()


    for i in range(1, len(all_data.T), 2):
        plt.scatter(all_data[:, i], all_data[:, i + 1])
        plt.show()
    exit()
    plt.scatter(all_data[:, 1], all_data[:, 3])
    plt.show()
    plt.scatter(all_data[:, 3], all_data[:, 5])
    plt.show()
    exit()
    for i in range(1, len(all_data.T)):
        plt.plot(kept_secs, all_data[:, i])
        plt.show()


def main_assemble():
    key = "PSol"
    key = "PAC"

    all_data = []
    first_day_offset = None
    for l in sys.stdin:
        pkl_filename = l.strip()
        all_daily_data, date_str = extract_data(pkl_filename, expected_module_count=6, key=key)

        full_date_str = "20" + date_str
        dt = datetime.strptime(full_date_str, "%Y%m%d")
        day_offset = int(dt.timestamp())

        if first_day_offset is None:
            first_day_offset = day_offset

        all_daily_data[:, 0] += day_offset
        all_data.append(all_daily_data)

    all_data = np.concatenate(all_data, axis=0)
    # all_data[:, 0] -= first_day_offset

    np.save("pac.npy", all_data)

    kept_secs = all_data[:, 0]
    # assert np.allclose(kept_secs, np.sort(kept_secs))

    aggregated = all_data[:, 1:].sum(axis=-1)

    gap_threshold = 60
    time_diff = np.diff(kept_secs)
    gap_indices = np.nonzero(time_diff > gap_threshold)[0]
    print("gap timestamps", kept_secs[gap_indices], time_diff[gap_indices])

    jumpback_indices = np.nonzero(time_diff < 0)[0]
    print("jumpback timestamps", kept_secs[jumpback_indices], time_diff[jumpback_indices])

    # kept_secs[1:][time_diff > gap_threshold] = np.nan
    # aggregated[1:][time_diff > gap_threshold] = np.nan

    plt.plot(kept_secs, aggregated, label="PSol aggregated")
    plt.scatter(kept_secs[gap_indices], np.zeros_like(kept_secs[gap_indices]), c="r", s=20, label="gap in records")
    plt.scatter(kept_secs[jumpback_indices], np.zeros_like(kept_secs[jumpback_indices]), c="orange", s=20, label="jumpback in records")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main_assemble()
