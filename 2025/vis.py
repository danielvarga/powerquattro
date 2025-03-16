import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import collections
from datetime import datetime

from base import *


def main():
    all_data = np.load("psol.npy")

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
    main()
