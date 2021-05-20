# Plot dataset CDF

import csv
from math import ceil
import matplotlib.pyplot as plt
from pathlib import Path

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'dataset_cdf'
capacity = 300

EMPIRICAL_DATA_FILEPATH = "./creditcard-non-fraudulent-only-amounts-only.csv"
with open(EMPIRICAL_DATA_FILEPATH, newline='') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    empirical_data = list(reader)
    empirical_data = [ceil(x[0]) for x in empirical_data if (0 < x[0] <= capacity)]  # Convert to float from list
    data_max = max(empirical_data)

    fig, ax = plt.subplots()
    # plt.xscale("log")
    n_bins = 1000
    # Plot the cumulative histogram
    n, bins, patches = plt.hist(empirical_data, n_bins, density=True, cumulative=True, label='CDF', histtype='step', alpha=1)

    ax.grid(True)
    ax.set_axisbelow(True)
    # ax.legend(loc='right')
    # ax.set_title('Cumulative step histogram')
    ax.set_xlabel('Amount')
    ax.set_ylabel('CDF')

    fig.savefig(save_at_directory + "dataset_cdf.png", bbox_inches='tight')
    fig.savefig(save_at_directory + "dataset_cdf.pdf", bbox_inches='tight')
    plt.show()
