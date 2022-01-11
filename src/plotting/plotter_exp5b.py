# Plot bar graph of success_rate vs transaction amount

from pypet import load_trajectory, pypetconstants
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

from save_legend import save_legend

outputs_directory = str(Path("../../outputs").resolve())
save_at_directory = outputs_directory + "/figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'results_111'
# filename = 'results_112'
# filename = 'results_113'
# filename = 'results_114'

traj = load_trajectory(filename=outputs_directory + '/HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename=outputs_directory + '/HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_parameters=2, load_results=1)
# traj.v_auto_load = True

# Parse parameter values
par_buffering_capability_values = traj.f_get('buffering_capability').f_get_range()
par_buffering_capability_values = list(dict.fromkeys(par_buffering_capability_values))
par_buffer_discipline_values = traj.f_get('buffer_discipline').f_get_range()
par_buffer_discipline_values = list(dict.fromkeys(par_buffer_discipline_values))
par_scheduling_policy_values = traj.f_get('scheduling_policy').f_get_range()
par_scheduling_policy_values = list(dict.fromkeys(par_scheduling_policy_values))
par_max_buffering_time_values = traj.f_get('max_buffering_time').f_get_range()
par_max_buffering_time_values = list(dict.fromkeys(par_max_buffering_time_values))
par_seed_values = traj.f_get('seed').f_get_range()
par_seed_values = list(dict.fromkeys(par_seed_values))

plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})

capacity = traj.f_get('capacity').f_get()
bin_count = 10
bin_edges = np.linspace(0, capacity, bin_count+1)

all_transactions_list_from_runs = traj.f_get_from_runs('all_transactions_list', fast_access=True)
experiment_anatomy_summary_dict = {}
experiment_anatomy_dict = {}

for index in all_transactions_list_from_runs:
    traj.v_idx = index
    succeeded_per_amount = {}
    total_per_amount = {}
    success_count_per_amount = {}
    atl = traj.crun.all_transactions_list.all_transactions_list
    atl = atl.to_dict('records')
    for t in atl:
        if t['amount'] in total_per_amount:
            total_per_amount[t['amount']] += 1
        else:
            total_per_amount[t['amount']] = 1
        if t['status'] == "SUCCEEDED":
            if t['amount'] in succeeded_per_amount:
                succeeded_per_amount[t['amount']] += 1
            else:
                succeeded_per_amount[t['amount']] = 1
    for amount in total_per_amount.keys():
        if amount in succeeded_per_amount.keys():
            success_count_per_amount[amount] = succeeded_per_amount[amount]
        else:
            success_count_per_amount[amount] = 0

    experiment_anatomy_dict[(traj.f_get('scheduling_policy').f_get(), traj.f_get('buffer_discipline').f_get(), traj.f_get('buffering_capability').f_get(), traj.f_get('max_buffering_time').f_get(), traj.f_get('seed').f_get())] = [success_count_per_amount, total_per_amount]
    # Contains a dictionary with all tx amounts as keys and the number of successful txs of each amount as values, for one group_index and one repetition


experiment_anatomy_in_bins_dict = {}
success_rate_of_every_bin_for_all_experiments = {}
average_success_rate_of_every_bin_for_all_experiments = {}
max_success_rate_of_every_bin_for_all_experiments = {}
min_success_rate_of_every_bin_for_all_experiments = {}

for scheduling_policy in par_scheduling_policy_values:
    for buffer_discipline in par_buffer_discipline_values:
        for buffering_capability in par_buffering_capability_values:
            for max_buffering_time in par_max_buffering_time_values:
                for seed in par_seed_values:
                    experiment_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time, seed)
                    success_count_per_amount = experiment_anatomy_dict[experiment_index][0]
                    total_per_amount = experiment_anatomy_dict[experiment_index][1]

                    # Group successful and total in bins for this experiment
                    successful_txs_in_bin = np.zeros(bin_count)
                    total_txs_in_bin = np.zeros(bin_count)

                    for amount in success_count_per_amount.keys():
                        if amount == bin_edges[bin_count]:
                            successful_txs_in_bin[bin_count-1] += success_count_per_amount[amount]
                        else:
                            for i in range(bin_count):  # could be done faster via binary search instead of linear search
                                if (bin_edges[i] <= amount) and (amount < bin_edges[i + 1]):
                                    successful_txs_in_bin[i] += success_count_per_amount[amount]

                    for amount in total_per_amount.keys():
                        if amount == bin_edges[bin_count]:
                            total_txs_in_bin[bin_count-1] += total_per_amount[amount]
                        else:
                            for i in range(bin_count):  # could be done faster via binary search instead of linear search
                                if (bin_edges[i] <= amount) and (amount < bin_edges[i + 1]):
                                    total_txs_in_bin[i] += total_per_amount[amount]

                    experiment_anatomy_in_bins_dict[experiment_index] = [successful_txs_in_bin, total_txs_in_bin]

                    # Calculate success rate for each bin for this experiment
                    success_rate_of_every_bin_for_experiment = [j / k if k else 0 for j, k in zip(successful_txs_in_bin, total_txs_in_bin)]
                    success_rate_of_every_bin_for_all_experiments[experiment_index] = success_rate_of_every_bin_for_experiment

                # Average success rate vectors over experiments
                group_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)
                # average_success_rate_of_every_bin_for_all_experiments[group_index] = np.zeros(bin_count)
                # for seed in par_seed_values:
                #     experiment_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time, seed)
                #     average_success_rate_of_every_bin_for_all_experiments[group_index] += success_rate_of_every_bin_for_all_experiments[experiment_index]
                # average_success_rate_of_every_bin_for_all_experiments[group_index] /= len(par_seed_values)

                average_success_rate_of_every_bin_for_all_experiments[group_index] = np.mean([success_rate_of_every_bin_for_all_experiments[group_index + (seed,)] for seed in par_seed_values], axis=0)
                max_success_rate_of_every_bin_for_all_experiments[group_index] = np.amax([success_rate_of_every_bin_for_all_experiments[group_index + (seed,)] for seed in par_seed_values], axis=0)
                min_success_rate_of_every_bin_for_all_experiments[group_index] = np.amin([success_rate_of_every_bin_for_all_experiments[group_index + (seed,)] for seed in par_seed_values], axis=0)


for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
    for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values):
        for buffering_capability_index, buffering_capability in enumerate(par_buffering_capability_values):
            fig, ax1 = plt.subplots()
            data_to_plot = []
            yerr = []
            for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
                group_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)
                data_to_plot.append([100*x for x in average_success_rate_of_every_bin_for_all_experiments[group_index]])
                yerr.append([100*(average_success_rate_of_every_bin_for_all_experiments[group_index] - min_success_rate_of_every_bin_for_all_experiments[group_index]), 100*(max_success_rate_of_every_bin_for_all_experiments[group_index] - average_success_rate_of_every_bin_for_all_experiments[group_index])])

            # ax1.hist(bin_edges[:bin_count-1], bins=bin_edges, weights=data_to_plot, label=par_scheduling_policy_values, color=colors[0:len(par_scheduling_policy_values)], alpha=1)
            n = len(par_scheduling_policy_values)
            bar_width = 1/(n+1)
            for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
                ax1.bar(np.arange(bin_count)-n*bar_width/2+scheduling_policy_index*bar_width, data_to_plot[scheduling_policy_index], yerr=yerr[scheduling_policy_index], label=par_scheduling_policy_values[scheduling_policy_index], color=colors[scheduling_policy_index], alpha=1, width=bar_width)

            ax1.set_ylim(bottom=0, top=100)
            ax1.set_xlabel("Transaction amount")
            ax1.set_ylabel("Success rate (%)")
            # my_xticklabels = []
            # for i in range(bin_count):
            #     my_xticklabels.append("["+str(int(bin_edges[i]))+","+str(int(bin_edges[i+1]))+"]")
            ax1.set_xticks(np.linspace(0-(n+2)*bar_width/2, bin_count-(n+2)*bar_width/2, 11))
            ax1.set_xticklabels([int(x) for x in bin_edges], rotation=0)
            ax1.grid(True)
            ax1.set_axisbelow(True)

            # plt.xscale("log")
            lines, labels = ax1.get_legend_handles_labels()
            legend = plt.legend(lines, labels, loc='best')
            plt.show()

            # plt.title("Success rate as a function of transaction amount")
            fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".png", bbox_inches='tight')
            fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".pdf", bbox_inches='tight')

            legend_filename = filename + "_legend.png"
            save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)