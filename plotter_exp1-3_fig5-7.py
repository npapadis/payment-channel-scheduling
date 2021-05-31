# Plot success_rate/throughput/sacrificed vs max buffering time for all scheduling policies

from math import ceil, floor

from matplotlib.lines import Line2D
from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

from save_legend import save_legend

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'results_101'
# filename = 'results_102'
# filename = 'results_103'
# filename = 'results_104'
# filename = 'results_105'
# filename = 'results_106'

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
normalized_throughput_channel_total_values = list(traj.f_get_from_runs('normalized_throughput_channel_total', fast_access=True).values())
normalized_throughput_channel_total_values = np.reshape(np.array(normalized_throughput_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
normalized_throughput_channel_total_values_average = normalized_throughput_channel_total_values.mean(axis=4)
normalized_throughput_channel_total_values_max = normalized_throughput_channel_total_values.max(axis=4)
normalized_throughput_channel_total_values_min = normalized_throughput_channel_total_values.min(axis=4)

# sacrificed_count_node_0_values = list(traj.f_get_from_runs('sacrificed_count_node_0', fast_access=True).values())
# sacrificed_count_node_0_values = np.reshape(np.array(sacrificed_count_node_0_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_node_0_values_average = sacrificed_count_node_0_values.mean(axis=4)
# sacrificed_count_node_0_values_max = sacrificed_count_node_0_values.max(axis=4)
# sacrificed_count_node_0_values_min = sacrificed_count_node_0_values.min(axis=4)
# sacrificed_count_node_1_values = list(traj.f_get_from_runs('sacrificed_count_node_1', fast_access=True).values())
# sacrificed_count_node_1_values = np.reshape(np.array(sacrificed_count_node_1_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_node_1_values_average = sacrificed_count_node_1_values.mean(axis=4)
# sacrificed_count_node_1_values_max = sacrificed_count_node_1_values.max(axis=4)
# sacrificed_count_node_1_values_min = sacrificed_count_node_1_values.min(axis=4)
# sacrificed_count_channel_total_values = list(traj.f_get_from_runs('sacrificed_count_channel_total', fast_access=True).values())
# sacrificed_count_channel_total_values = np.reshape(np.array(sacrificed_count_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_count_channel_total_values_average = sacrificed_count_channel_total_values.mean(axis=4)
# sacrificed_count_channel_total_values_max = sacrificed_count_channel_total_values.max(axis=4)
# sacrificed_count_channel_total_values_min = sacrificed_count_channel_total_values.min(axis=4)


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})
markers = [".", "x", "s", "+", "*", "d"]



for buffering_capability_index, buffering_capability in enumerate(par_buffering_capability_values):
    for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
        fig, ax1 = plt.subplots()
        # ax1.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
        # ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        # ax3 = ax1.twinx()  # instantiate a third axis that shares the same x-axis

        for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
            innermost_index = scheduling_policy_index
            color = colors[innermost_index]
            # # Success rates
            # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 0, Scheduling policy: "+scheduling_policy, linestyle='solid')
            # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 1, Scheduling policy: "+scheduling_policy, linestyle='solid')
            # ax1.plot(par_max_buffering_time_values, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate total, Scheduling policy: "+scheduling_policy, linestyle=linestyles[0], color=color, alpha=1)
            # ax2.plot(par_max_buffering_time_values, success_amount_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 0, Scheduling policy: "+scheduling_policy, linestyle='dashed')
            # ax2.plot(par_max_buffering_time_values, success_amount_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 1, Scheduling policy: "+scheduling_policy, linestyle='dashed')
            # ax2.plot(par_max_buffering_time_values, success_amount_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Total successful amount, Scheduling policy: "+scheduling_policy, linestyle=linestyles[1], color=color, alpha=1)

            # Total throughput
            ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy, linestyle=linestyles[0], marker=markers[innermost_index], color=color, alpha=1)
            yerr_total = [100*(normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
            ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_total, fmt='none')

            # # Per-node throughputs
            # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy, linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
            # yerr_0 = [100*(normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_0_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
            # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_0, color=color, fmt='none')
            # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)
            # yerr_1 = [100*(normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_1_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
            # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_1, color=color, fmt='none')

            # # Per-node sacrificed
            # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy, linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
            # yerr_0 = [sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - sacrificed_count_node_0_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], sacrificed_count_node_0_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]]
            # ax1.errorbar(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_0, color=color, fmt='none')
            # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)
            # yerr_1 = [sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - sacrificed_count_node_1_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], sacrificed_count_node_1_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]]
            # ax1.errorbar(par_max_buffering_time_values, sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_1, color=color, fmt='none')

            # ax3.plot(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] + sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Number of sacrificed transactions, Scheduling policy: "+scheduling_policy, linestyle=linestyles[3], color=color, alpha=1)

        ax1.grid(True)
        ax1.set_ylim(bottom=0, top=100)
        # ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Maximum buffering time (sec)")
        # ax1.set_ylabel("Success rate (%)")
        ax1.set_ylabel("Normalized throughput (%)")
        # ax1.set_ylabel("Number of sacrificed transactions")
        # # ax2.set_ylabel("Successful amount (coins)")
        # ax2.tick_params(axis='y', rotation=45)
        # ax3.set_ylabel("Number of sacrificed transactions")
        # ax3.spines["right"].set_position(("axes", 1.2))
        # plt.title("Normalized throughput as a function of the maximum buffering time")

        # # Per-node throughput/sacrificed legend
        # lines_1, labels_1 = ax1.get_legend_handles_labels()
        # for h in lines_1: h.set_linestyle("")
        # lines_2 = [Line2D([0], [0], color='k', linewidth=1, linestyle=linestyles[1]),
        #            Line2D([0], [0], color='k', linewidth=1, linestyle=linestyles[3])]
        # labels_2 = ["Node A", "Node B"]
        # lines = lines_1 + lines_2
        # labels = labels_1 + labels_2
        # legend = ax1.legend(lines, labels, loc='best')
        # for h in lines_1: h.set_linestyle(linestyles[1])

        # Total throughput legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # lines_3, labels_3 = ax3.get_legend_handles_labels()
        # lines = lines_1 + lines_2 + lines_3
        # labels = labels_1 + labels_2 + labels_3
        lines = lines_1
        labels = labels_1
        legend = ax1.legend(lines, labels, loc='best')



        fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".png", bbox_inches='tight')
        fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + ".pdf", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_nodes.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_nodes.pdf", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_sac.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_" + str(buffer_discipline_index+1) + "_sac.pdf", bbox_inches='tight')

        # lines_1, labels_1 = ax1.get_legend_handles_labels()
        # # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # # lines_3, labels_3 = ax3.get_legend_handles_labels()
        # # lines = lines_1 + lines_2 + lines_3
        # # labels = labels_1 + labels_2 + labels_3
        # lines = lines_1
        # labels = labels_1
        # legend = plt.legend(lines, labels)
        # legend_filename = filename + "_legend.png"
        # save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

        # handles, labels = ax1.get_legend_handles_labels()
        # axe.legend(handles, labels, loc=loc)
        # axe.xaxis.set_visible(False)
        # axe.yaxis.set_visible(False)
        # for v in axe.spines.values():
        #     v.set_visible(False)

plt.show()
