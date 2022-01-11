import csv
from math import ceil, floor
from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'results_114'


def save_legend(fig, lines, labels, legend, legend_directory, legend_filename):
    # Code modified from https://gist.github.com/rldotai/012dbb3294bb2599ff82e61e82356990

    # ---------------------------------------------------------------------
    # Create a separate figure for the legend
    # ---------------------------------------------------------------------
    # Bounding box for legend (in order to get proportions right)
    # Issuing a draw command seems necessary in order to ensure the layout
    # happens correctly; I was getting weird bounding boxes without it.
    # fig.canvas.draw()
    # This gives pixel coordinates for the legend bbox (although perhaps
    # if you are using a different renderer you might get something else).
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    # Convert pixel coordinates to inches
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())

    # Create the separate figure, with appropriate width and height
    # Create a separate figure for the legend
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))

    # Recreate the legend on the separate figure/axis
    legend_squared = legend_ax.legend(
        # *ax.get_legend_handles_labels(),
        lines,
        labels,
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=False,
        fancybox=None,
        shadow=False,
        # ncol=legend_cols,
        mode='expand',
    )

    # Remove everything else from the legend's figure
    legend_ax.axis('off')

    # Save the legend as a separate figure
    # print(f"Saving to: {legend_fullpath}")

    Path(legend_directory).mkdir(parents=True, exist_ok=True)
    legend_fullpath = legend_directory + "/" + legend_filename
    legend_fig.savefig(
        legend_fullpath,
        bbox_inches='tight',
        bbox_extra_artists=[legend_squared],
    )


# # Plot dataset CDF
# capacity = 300
# EMPIRICAL_DATA_FILEPATH = "./creditcard-non-fraudulent-only-amounts-only.csv"
# with open(EMPIRICAL_DATA_FILEPATH, newline='') as f:
#     reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
#     empirical_data = list(reader)
#     empirical_data = [ceil(x[0]) for x in empirical_data if (0 < x[0] <= capacity)]  # Convert to float from list
#     data_max = max(empirical_data)
#
#     fig, ax = plt.subplots()
#     # plt.xscale("log")
#     n_bins = 1000
#     # plot the cumulative histogram
#     n, bins, patches = plt.hist(empirical_data, n_bins, density=True, cumulative=True, label='CDF', histtype='step', alpha=1)
#
#     ax.grid(True)
#     # ax.legend(loc='right')
#     # ax.set_title('Cumulative step histogram')
#     ax.set_xlabel('Amount')
#     ax.set_ylabel('CDF')
#
#     fig.savefig(save_at_directory + "dataset_cdf.png", bbox_inches='tight')
#     fig.savefig(save_at_directory + "dataset_cdf.eps", bbox_inches='tight')
#     plt.show()
# exit()



traj = load_trajectory(filename='./HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_parameters=2, load_results=1)
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



# Parse results

# success_rate_node_0_values = list(traj.f_get_from_runs('success_rate_node_0', fast_access=True).values())
# success_rate_node_0_values = np.reshape(np.array(success_rate_node_0_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_rate_node_0_values_average = success_rate_node_0_values.mean(axis=4)
# success_rate_node_1_values = list(traj.f_get_from_runs('success_rate_node_1', fast_access=True).values())
# success_rate_node_1_values = np.reshape(np.array(success_rate_node_1_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_rate_node_1_values_average = success_rate_node_1_values.mean(axis=4)
# success_rate_channel_total_values = list(traj.f_get_from_runs('success_rate_channel_total', fast_access=True).values())
# success_rate_channel_total_values = np.reshape(np.array(success_rate_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_rate_channel_total_values_average = success_rate_channel_total_values.mean(axis=4)
# success_rate_channel_total_values_max = success_rate_channel_total_values.max(axis=4)
# success_rate_channel_total_values_min = success_rate_channel_total_values.min(axis=4)
#
# success_amount_node_0_values = list(traj.f_get_from_runs('success_amount_node_0', fast_access=True).values())
# success_amount_node_0_values = np.reshape(np.array(success_amount_node_0_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_amount_node_0_values_average = success_amount_node_0_values.mean(axis=4)
# success_amount_node_1_values = list(traj.f_get_from_runs('success_amount_node_1', fast_access=True).values())
# success_amount_node_1_values = np.reshape(np.array(success_amount_node_1_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_amount_node_1_values_average = success_amount_node_1_values.mean(axis=4)
# success_amount_channel_total_values = list(traj.f_get_from_runs('success_amount_channel_total', fast_access=True).values())
# success_amount_channel_total_values = np.reshape(np.array(success_amount_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_amount_channel_total_values_average = success_amount_channel_total_values.mean(axis=4)
#
# normalized_throughput_node_0_values = list(traj.f_get_from_runs('normalized_throughput_node_0', fast_access=True).values())
# normalized_throughput_node_0_values = np.reshape(np.array(normalized_throughput_node_0_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# normalized_throughput_node_0_values_average = normalized_throughput_node_0_values.mean(axis=4)
# normalized_throughput_node_0_values_max = normalized_throughput_node_0_values.max(axis=4)
# normalized_throughput_node_0_values_min = normalized_throughput_node_0_values.min(axis=4)
# normalized_throughput_node_1_values = list(traj.f_get_from_runs('normalized_throughput_node_1', fast_access=True).values())
# normalized_throughput_node_1_values = np.reshape(np.array(normalized_throughput_node_1_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# normalized_throughput_node_1_values_average = normalized_throughput_node_1_values.mean(axis=4)
# normalized_throughput_node_1_values_max = normalized_throughput_node_1_values.max(axis=4)
# normalized_throughput_node_1_values_min = normalized_throughput_node_1_values.min(axis=4)
# normalized_throughput_channel_total_values = list(traj.f_get_from_runs('normalized_throughput_channel_total', fast_access=True).values())
# normalized_throughput_channel_total_values = np.reshape(np.array(normalized_throughput_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# normalized_throughput_channel_total_values_average = normalized_throughput_channel_total_values.mean(axis=4)
# normalized_throughput_channel_total_values_max = normalized_throughput_channel_total_values.max(axis=4)
# normalized_throughput_channel_total_values_min = normalized_throughput_channel_total_values.min(axis=4)
#
# sacrificed_count_node_0_values = list(traj.f_get_from_runs('sacrificed_count_node_0', fast_access=True).values())
# sacrificed_count_node_0_values = np.reshape(np.array(sacrificed_count_node_0_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_node_0_values_average = sacrificed_count_node_0_values.mean(axis=4)
# sacrificed_count_node_1_values = list(traj.f_get_from_runs('sacrificed_count_node_1', fast_access=True).values())
# sacrificed_count_node_1_values = np.reshape(np.array(sacrificed_count_node_1_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_node_1_values_average = sacrificed_count_node_1_values.mean(axis=4)
# sacrificed_count_channel_total_values = list(traj.f_get_from_runs('sacrificed_count_channel_total', fast_access=True).values())
# sacrificed_count_channel_total_values = np.reshape(np.array(sacrificed_count_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_channel_total_values_average = sacrificed_count_channel_total_values.mean(axis=4)


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# plt.rcParams['font.size'] = '10'
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})
# colors = cycler(color='bgrcmky')
markers = [".", "x", "s", "+", "*", "d"]


# # =========================================================================================
#
#
# # Success_rate/throughput/sacrificed vs max buffering time for all scheduling policies
# for buffering_capability_index, buffering_capability in enumerate(par_buffering_capability_values):
#     for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
#         fig, ax1 = plt.subplots()
#         # ax1.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
#         # ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
#         # ax3 = ax1.twinx()  # instantiate a third axis that shares the same x-axis
#
#         for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#             innermost_index = scheduling_policy_index
#             color = colors[innermost_index]
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 0, Scheduling policy: "+scheduling_policy, linestyle='solid')
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 1, Scheduling policy: "+scheduling_policy, linestyle='solid')
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate total, Scheduling policy: "+scheduling_policy, linestyle=linestyles[0], color=color, alpha=1)
#             # ax2.plot(par_max_buffering_time_values, success_amount_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 0, Scheduling policy: "+scheduling_policy, linestyle='dashed')
#             # ax2.plot(par_max_buffering_time_values, success_amount_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 1, Scheduling policy: "+scheduling_policy, linestyle='dashed')
#             # ax2.plot(par_max_buffering_time_values, success_amount_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Total successful amount, Scheduling policy: "+scheduling_policy, linestyle=linestyles[1], color=color, alpha=1)
#
#             # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node A", linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
#             # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node B", linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)
#             # yerr_0 = [100*(normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_0_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
#             # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_0, fmt='none')
#             # yerr_1 = [100*(normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_1_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
#             # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_1, fmt='none')
#
#             ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy, linestyle=linestyles[0], marker=markers[innermost_index], color=color, alpha=1)
#             yerr_total = [100*(normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
#             ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_total, fmt='none')
#
#             # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node A", linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
#             # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node B", linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)
#
#
#             # ax3.plot(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] + sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Number of sacrificed transactions, Scheduling policy: "+scheduling_policy, linestyle=linestyles[3], color=color, alpha=1)
#
#         ax1.grid(True)
#         ax1.set_ylim(bottom=0, top=100)
#         # ax1.set_ylim(bottom=0)
#         ax1.set_xlabel("Maximum buffering time (sec)")
#         # ax1.set_ylabel("Success rate (%)")
#         # ax1.set_ylabel("Normalized throughput (%)")
#         ax1.set_ylabel("Number of sacrificed transactions")
#         # # ax2.set_ylabel("Successful amount (coins)")
#         # ax2.tick_params(axis='y', rotation=45)
#         # ax3.set_ylabel("Number of sacrificed transactions")
#         # ax3.spines["right"].set_position(("axes", 1.2))
#         # plt.title("Normalized throughput as a function of the maximum buffering time")
#
#         lines_1, labels_1 = ax1.get_legend_handles_labels()
#         # lines_2, labels_2 = ax2.get_legend_handles_labels()
#         # lines_3, labels_3 = ax3.get_legend_handles_labels()
#         # lines = lines_1 + lines_2 + lines_3
#         # labels = labels_1 + labels_2 + labels_3
#         lines = lines_1
#         labels = labels_1
#         legend = ax1.legend(lines, labels, loc='best')
#
#
#         # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".png", bbox_inches='tight')
#         # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".eps", bbox_inches='tight')
#         # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_nodes.png", bbox_inches='tight')
#         # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_nodes.eps", bbox_inches='tight')
#         fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_sac.png", bbox_inches='tight')
#         fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_sac.eps", bbox_inches='tight')
#
#         lines_1, labels_1 = ax1.get_legend_handles_labels()
#         # lines_2, labels_2 = ax2.get_legend_handles_labels()
#         # lines_3, labels_3 = ax3.get_legend_handles_labels()
#         # lines = lines_1 + lines_2 + lines_3
#         # labels = labels_1 + labels_2 + labels_3
#         lines = lines_1
#         labels = labels_1
#         legend = plt.legend(lines, labels)
#         legend_filename = filename + "_legend.png"
#         save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)
#
#         # handles, labels = ax1.get_legend_handles_labels()
#         # axe.legend(handles, labels, loc=loc)
#         # axe.xaxis.set_visible(False)
#         # axe.yaxis.set_visible(False)
#         # for v in axe.spines.values():
#         #     v.set_visible(False)
#
# # plt.show()
#
# exit()
#
#
# # =========================================================================================
#
#
# # whb experiments
# for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
#     for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#         for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values):
#             fig, ax1 = plt.subplots()
#             ax1.bar(par_buffering_capability_values, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], label="Success rate", color=colors[0])
#             # ax1.bar(par_buffering_capability_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], label="Normalized throughput", color=colors[1])
#
#             ax1.grid(True)
#             ax1.set_ylim(bottom=0, top=100)
#             ax1.set_xlabel("Who has a buffer")
#             ax1.set_ylabel("Success rate (%)")
#             # ax1.set_ylabel("Normalized throughput (%)")
#             lines_1, labels_1 = ax1.get_legend_handles_labels()
#             lines = lines_1
#             labels = labels_1
#             fig.savefig(save_at_directory + filename + "_" + labels_1[0] + "_sr.png", bbox_inches='tight')
#             fig.savefig(save_at_directory + filename + "_" + labels_1[0] + "_sr.eps", bbox_inches='tight')
#             # fig.savefig(save_at_directory + filename + "_" + labels_1[0] + "_nthr.png", bbox_inches='tight')
#             # fig.savefig(save_at_directory + filename + "_" + labels_1[0] + "_nthr.eps", bbox_inches='tight')
#             legend = plt.legend(lines, labels)
#             legend_filename = filename + "_legend.png"
#             save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)
#
# exit()


# =========================================================================================

capacity = traj.f_get('capacity').f_get()
bin_count = 10
bin_edges = np.linspace(0, capacity, bin_count+1)

# Success_rate vs transaction amount

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
                            for i in range(bin_count - 1):  # could be done faster via binary search instead of linear search
                                if (bin_edges[i] <= amount) and (amount < bin_edges[i + 1]):
                                    successful_txs_in_bin[i] += success_count_per_amount[amount]

                    for amount in total_per_amount.keys():
                        if amount == bin_edges[bin_count]:
                            total_txs_in_bin[bin_count-1] += total_per_amount[amount]
                        else:
                            for i in range(bin_count - 1):  # could be done faster via binary search instead of linear search
                                if (bin_edges[i] <= amount) and (amount < bin_edges[i + 1]):
                                    total_txs_in_bin[i] += total_per_amount[amount]

                    experiment_anatomy_in_bins_dict[experiment_index] = [successful_txs_in_bin, total_txs_in_bin]

                    # Calculate success rate for each bin for this experiment
                    success_rate_of_every_bin_for_experiment = [j / k if k else 0 for j, k in zip(successful_txs_in_bin, total_txs_in_bin)]
                    success_rate_of_every_bin_for_all_experiments[experiment_index] = success_rate_of_every_bin_for_experiment

                # Average success rate vectors over experiments
                group_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)
                average_success_rate_of_every_bin_for_all_experiments[group_index] = np.zeros(bin_count)
                for seed in par_seed_values:
                    experiment_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time, seed)
                    average_success_rate_of_every_bin_for_all_experiments[group_index] += success_rate_of_every_bin_for_all_experiments[experiment_index]
                average_success_rate_of_every_bin_for_all_experiments[group_index] /= len(par_seed_values)


for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
    for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values):
        for buffering_capability_index, buffering_capability in enumerate(par_buffering_capability_values):
            fig, ax1 = plt.subplots()
            data_to_plot = []
            for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
                group_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)
                data_to_plot.append([100*x for x in average_success_rate_of_every_bin_for_all_experiments[group_index]])

            # ax1.hist(bin_edges[:bin_count-1], bins=bin_edges, weights=data_to_plot, label=par_scheduling_policy_values, color=colors[0:len(par_scheduling_policy_values)], alpha=1)
            n = len(par_scheduling_policy_values)
            bar_width = 1/(n+1)
            for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
                ax1.bar(np.arange(bin_count)-n*bar_width/2+scheduling_policy_index*bar_width, data_to_plot[scheduling_policy_index], label=par_scheduling_policy_values[scheduling_policy_index], color=colors[scheduling_policy_index], alpha=1, width=bar_width)

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
            fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".eps", bbox_inches='tight')
#
#             # lines, labels = ax1.get_legend_handles_labels()
#             # legend = plt.legend(lines, labels)
#             # legend_filename = filename + "_legend.png"
#             # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

exit()


# =========================================================================================



# Success_rate vs transaction deadline
all_transactions_list_from_runs = traj.f_get_from_runs('all_transactions_list', fast_access=True)
experiment_anatomy_summary_dict = {}

for scheduling_policy in par_scheduling_policy_values:
    experiment_anatomy_summary_dict[scheduling_policy] = {}
    for buffer_discipline in par_buffer_discipline_values:
        for buffering_capability in par_buffering_capability_values:
            for max_buffering_time in par_max_buffering_time_values:

                experiment_anatomy_dict = {}

                for index in all_transactions_list_from_runs:
                    traj.v_idx = index
                    succeeded_per_deadline = {}
                    total_per_deadline = {}
                    success_count_per_deadline = {}
                    atl = traj.crun.all_transactions_list.all_transactions_list
                    atl = atl.to_dict('records')
                    for t in atl:
                        if t['max_buffering_time'] in total_per_deadline:
                            total_per_deadline[t['max_buffering_time']] += 1
                        else:
                            total_per_deadline[t['max_buffering_time']] = 1
                        if t['status'] == "SUCCEEDED":
                            if t['max_buffering_time'] in succeeded_per_deadline:
                                succeeded_per_deadline[t['max_buffering_time']] += 1
                            else:
                                succeeded_per_deadline[t['max_buffering_time']] = 1
                    for deadline in total_per_deadline.keys():
                        if deadline in succeeded_per_deadline.keys():
                            success_count_per_deadline[deadline] = succeeded_per_deadline[deadline]
                        else:
                            success_count_per_deadline[deadline] = 0

                    experiment_anatomy_dict[(traj.scheduling_policy, traj.buffer_discipline, traj.buffering_capability, traj.max_buffering_time, traj.seed)] = success_count_per_deadline


                group_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)
                experiment_anatomy_summary_dict[scheduling_policy][group_index] = {}
                for seed in par_seed_values:
                    experiment_index = (scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time, seed)
                    for deadline in experiment_anatomy_dict[experiment_index].keys():
                        if deadline in experiment_anatomy_summary_dict[scheduling_policy][group_index].keys():
                            experiment_anatomy_summary_dict[scheduling_policy][group_index][deadline] += experiment_anatomy_dict[experiment_index][deadline]
                        else:
                            experiment_anatomy_summary_dict[scheduling_policy][group_index][deadline] = experiment_anatomy_dict[experiment_index][deadline]
                for deadline in experiment_anatomy_summary_dict[scheduling_policy][group_index].keys():
                    experiment_anatomy_summary_dict[scheduling_policy][group_index][deadline] /= len(par_seed_values)
                # Sort by amount
                experiment_anatomy_summary_dict[scheduling_policy][group_index] = dict(sorted(experiment_anatomy_summary_dict[scheduling_policy][group_index].items()))

mbt_step = min(len(par_max_buffering_time_values), max(4, (len(par_max_buffering_time_values)//4)))
bin_count = 10

for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
    for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values[::mbt_step]):
        for buffering_capability_index, buffering_capability in enumerate(par_buffering_capability_values):
            fig, ax1 = plt.subplots()
            all_deadlines_to_process = []
            all_weights = []
            for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
                innermost_index = scheduling_policy_index
                color = colors[innermost_index]

                deadlines_to_process = list(experiment_anatomy_summary_dict[scheduling_policy][(scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)].keys())
                success_counts_to_process = list(experiment_anatomy_summary_dict[scheduling_policy][(scheduling_policy, buffer_discipline, buffering_capability, max_buffering_time)].values())
                # bin_width = max(np.floor(max(amounts_to_process)) // bin_count)
                # amounts_to_plot, bin_edges = np.histogram(amounts_to_process, bins=bin_count, weights=success_counts_to_process)
                elements_per_bin, bin_edges = np.histogram(deadlines_to_process, bins=bin_count)
                # plt.bar(amounts_to_plot_not_normalized, amounts_to_plot_not_normalized / elements_per_bin, width=0.8)

                # weights = np.zeros(len(amounts_to_process))
                # for amount_index, amount in enumerate(amounts_to_process):
                #     bin_index = 0
                #     while amount > bin_edges[bin_index+1]:
                #         bin_index += 1
                #     # weights[amount_index] = 1 / elements_per_bin[bin_index]
                #     print("amount_index = ", amount_index)
                #     print("amount = ", amount)
                #     print("bin = ", bin_index, "\n")
                #     weights[amount_index] = success_counts_to_process[amount_index] / elements_per_bin[bin_index]

                bin_from_tx_index = np.digitize(deadlines_to_process, bin_edges)
                weights = [success_counts_to_process[i] / bin_from_tx_index[b] for i, b in enumerate(bin_from_tx_index)]
                # ax1.hist(amounts_to_process, weights=weights, bins=bin_count, label=scheduling_policy, color=color, alpha=0.5)
                all_deadlines_to_process.append(deadlines_to_process)
                all_weights.append(weights)

                # ax1.hist(amounts_to_process, weights=weights, bins=bin_count, label="Success rate, Who has buffer: "+buffering_capability, color=color)
                # plt.hist(amounts_to_process, weights=weights, bins=bin_count, label="Success rate, Who has buffer: "+buffering_capability, color=color)

                # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 0, Buffers: "+buffering_capability, linestyle='solid')
                # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 1, Buffers: "+buffering_capability, linestyle='solid')
                # ax1.plot(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+buffering_capability, linestyle=linestyles[0], color=color)
                # ax1.bar(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+buffering_capability, linestyle=linestyles[0], color=color)
                # ax1.scatter(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+buffering_capability, marker=markers[innermost_index], color=color)

            ax1.hist(all_deadlines_to_process, weights=all_weights, bins=bin_count, label=par_scheduling_policy_values, color=colors[0:len(par_scheduling_policy_values)], alpha=1)


            ax1.set_ylim(bottom=0, top=100)
            ax1.set_xlabel("Transaction maximum buffering time")
            ax1.set_ylabel("Success rate (%)")
            my_xticks = [int(x) for x in bin_edges]
            my_xticks[0] = 0
            ax1.set_xticks(my_xticks)
            # plt.xscale("log")
            lines, labels = ax1.get_legend_handles_labels()
            legend = plt.legend(lines, labels, loc='best')

            # plt.title("Success rate as a function of transaction amount")
            fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".png", bbox_inches='tight')

            # lines, labels = ax1.get_legend_handles_labels()
            # legend = plt.legend(lines, labels)
            # legend_filename = filename + "_legend.png"
            # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)
# plt.show()
