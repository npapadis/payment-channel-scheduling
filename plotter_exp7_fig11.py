# Plot success_rate/throughput/sacrificed vs max buffering time for all scheduling policies

from math import ceil, floor
from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
from pathlib import Path

from save_legend import save_legend

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'results_131'

traj = load_trajectory(filename='./HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/' + filename + '.hdf5', name='single_payment_channel_scheduling', load_parameters=2, load_results=1)
# traj.v_auto_load = True

# Parse parameter values (different parameters than in other experiments)

par_buffer_discipline_values = traj.f_get('buffer_discipline').f_get_range()
par_buffer_discipline_values = list(dict.fromkeys(par_buffer_discipline_values))
par_seed_values = traj.f_get('seed').f_get_range()
par_seed_values = list(dict.fromkeys(par_seed_values))
par_deadline_fraction_values = traj.f_get('deadline_fraction').f_get_range()
par_deadline_fraction_values = list(dict.fromkeys(par_deadline_fraction_values))


# Parse results

normalized_throughput_channel_total_values = list(traj.f_get_from_runs('normalized_throughput_channel_total', fast_access=True).values())
normalized_throughput_channel_total_values = np.reshape(np.array(normalized_throughput_channel_total_values), (len(par_buffer_discipline_values), len(par_seed_values), len(par_deadline_fraction_values)))
normalized_throughput_channel_total_values_average = normalized_throughput_channel_total_values.mean(axis=1)
normalized_throughput_channel_total_values_max = normalized_throughput_channel_total_values.max(axis=1)
normalized_throughput_channel_total_values_min = normalized_throughput_channel_total_values.min(axis=1)

# average_total_queueing_time_per_successful_unit_amount_values = list(traj.f_get_from_runs('average_total_queueing_time_per_successful_unit_amount', fast_access=True).values())
# average_total_queueing_time_per_successful_unit_amount_values = np.reshape(np.array(average_total_queueing_time_per_successful_unit_amount_values), (len(par_buffer_discipline_values), len(par_seed_values), len(par_deadline_fraction_values)))
# average_total_queueing_time_per_successful_unit_amount_values_average = average_total_queueing_time_per_successful_unit_amount_values.mean(axis=1)
# average_total_queueing_time_per_successful_unit_amount_values_max = average_total_queueing_time_per_successful_unit_amount_values.max(axis=1)
# average_total_queueing_time_per_successful_unit_amount_values_min = average_total_queueing_time_per_successful_unit_amount_values.min(axis=1)

average_total_queueing_time_per_successful_transaction_values = list(traj.f_get_from_runs('average_total_queueing_time_per_successful_transaction', fast_access=True).values())
average_total_queueing_time_per_successful_transaction_values = np.reshape(np.array(average_total_queueing_time_per_successful_transaction_values), (len(par_buffer_discipline_values), len(par_seed_values), len(par_deadline_fraction_values)))
average_total_queueing_time_per_successful_transaction_values_average = average_total_queueing_time_per_successful_transaction_values.mean(axis=1)
average_total_queueing_time_per_successful_transaction_values_max = average_total_queueing_time_per_successful_transaction_values.max(axis=1)
average_total_queueing_time_per_successful_transaction_values_min = average_total_queueing_time_per_successful_transaction_values.min(axis=1)


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})
markers = [".", "x", "s", "+", "*", "d"]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis

for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
    # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 0, Scheduling policy: "+scheduling_policy, linestyle='solid')
    # ax1.plot(par_max_buffering_time_values, 100 * success_rate_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate of node 1, Scheduling policy: "+scheduling_policy, linestyle='solid')
    # ax1.plot(par_max_buffering_time_values, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Success rate total, Scheduling policy: "+scheduling_policy, linestyle=linestyles[0], color=color, alpha=1)
    # ax2.plot(par_max_buffering_time_values, success_amount_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 0, Scheduling policy: "+scheduling_policy, linestyle='dashed')
    # ax2.plot(par_max_buffering_time_values, success_amount_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Throughput of node 1, Scheduling policy: "+scheduling_policy, linestyle='dashed')
    # ax2.plot(par_max_buffering_time_values, success_amount_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label="Total successful amount, Scheduling policy: "+scheduling_policy, linestyle=linestyles[1], color=color, alpha=1)

    # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node A", linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
    # ax1.plot(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node B", linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)
    # yerr_0 = [100*(normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_0_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
    # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_0, fmt='--')
    # yerr_1 = [100*(normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_min[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :]), 100*(normalized_throughput_node_1_values_max[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :] - normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :])]
    # ax1.errorbar(par_max_buffering_time_values, 100 * normalized_throughput_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], yerr=yerr_1, fmt='--')

    ax1.plot(par_deadline_fraction_values, 100 * normalized_throughput_channel_total_values_average[buffer_discipline_index, :], label=buffer_discipline, linestyle=linestyles[0], marker=markers[buffer_discipline_index], color=colors[buffer_discipline_index], alpha=1)
    yerr_total_1 = [100*(normalized_throughput_channel_total_values_average[buffer_discipline_index, :] - normalized_throughput_channel_total_values_min[buffer_discipline_index, :]), 100*(normalized_throughput_channel_total_values_max[buffer_discipline_index, :] - normalized_throughput_channel_total_values_average[buffer_discipline_index, :])]
    ax1.errorbar(par_deadline_fraction_values, 100 * normalized_throughput_channel_total_values_average[buffer_discipline_index, :], yerr=yerr_total_1, color=colors[0], fmt='none')

    # ax2.plot(par_deadline_fraction_values, average_total_queueing_time_per_successful_unit_amount_values_average[buffer_discipline_index, :], label="Queueing delay: " + buffer_discipline, linestyle=linestyles[1], marker=markers[buffer_discipline_index], color=colors[buffer_discipline_index], alpha=1)
    # yerr_total_2 = [average_total_queueing_time_per_successful_unit_amount_values_average[buffer_discipline_index, :] - average_total_queueing_time_per_successful_unit_amount_values_min[buffer_discipline_index, :], average_total_queueing_time_per_successful_unit_amount_values_max[buffer_discipline_index, :] - average_total_queueing_time_per_successful_unit_amount_values_average[buffer_discipline_index, :]]
    # ax2.errorbar(par_deadline_fraction_values, average_total_queueing_time_per_successful_unit_amount_values_average[buffer_discipline_index, :], yerr=yerr_total_2, color=colors[1], fmt='none')

    ax2.plot(par_deadline_fraction_values, average_total_queueing_time_per_successful_transaction_values_average[buffer_discipline_index, :], label="Queueing delay: " + buffer_discipline, linestyle=linestyles[1], marker=markers[buffer_discipline_index], color=colors[buffer_discipline_index], alpha=1)
    yerr_total_2 = [average_total_queueing_time_per_successful_transaction_values_average[buffer_discipline_index, :] - average_total_queueing_time_per_successful_transaction_values_min[buffer_discipline_index, :], average_total_queueing_time_per_successful_transaction_values_max[buffer_discipline_index, :] - average_total_queueing_time_per_successful_transaction_values_average[buffer_discipline_index, :]]
    ax2.errorbar(par_deadline_fraction_values, average_total_queueing_time_per_successful_transaction_values_average[buffer_discipline_index, :], yerr=yerr_total_2, color=colors[1], fmt='none')

    # ax1.hlines(45, 0, 1, color=colors[0])

    # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_0_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node A", linestyle=linestyles[1], marker=markers[innermost_index], color=color, alpha=1)
    # ax1.plot(par_max_buffering_time_values, sacrificed_count_node_1_values_average[scheduling_policy_index, buffer_discipline_index, buffering_capability_index, :], label=scheduling_policy+" - Node B", linestyle=linestyles[3], marker=markers[innermost_index], color=color, alpha=1)

ax1.grid(True)
ax1.set_ylim(bottom=0, top=100)
ax2.set_ylim(bottom=0)
ax1.set_xlabel("Fraction of deadline when PMDE is applied")
# ax1.set_ylabel("Success rate (%)")
ax1.set_ylabel("Normalized throughput (%)")
# ax1.set_ylabel("Number of sacrificed transactions")
# ax2.set_ylabel("Average queueing delay (sec)\n per successful unit amount")
ax2.set_ylabel("Average queueing delay (sec)\n per successful transaction")
# ax2.tick_params(axis='y', rotation=45)
# plt.title("")

fig.savefig(save_at_directory + filename + ".png", bbox_inches='tight')
fig.savefig(save_at_directory + filename + ".pdf", bbox_inches='tight')

lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()

for h in lines_1: h.set_linestyle("")
lines_2 = [Line2D([0], [0], color='k', linewidth=1, linestyle='-'), Line2D([0], [0], color='k', linewidth=1, linestyle='--')]
labels_2 = ["Normalized throughput", "Queueing delay"]

lines = lines_1 + lines_2
labels = labels_1 + labels_2
legend = ax1.legend(lines, labels, loc='best')
legend_filename = filename + "_legend.pdf"
save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)

# handles, labels = ax1.get_legend_handles_labels()
# axe.legend(handles, labels, loc=loc)
# axe.xaxis.set_visible(False)
# axe.yaxis.set_visible(False)
# for v in axe.spines.values():
#     v.set_visible(False)


# fig2, ax = plt.subplots()
# for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
#     ax.plot(average_total_queueing_time_per_successful_transaction_values_average[buffer_discipline_index, :], 100 * normalized_throughput_channel_total_values_average[buffer_discipline_index, :], label=buffer_discipline, linestyle=linestyles[0], marker=markers[buffer_discipline_index], color=colors[buffer_discipline_index], alpha=1)
#     # yerr_total_1 = [100*(normalized_throughput_channel_total_values_average[buffer_discipline_index, :] - normalized_throughput_channel_total_values_min[buffer_discipline_index, :]), 100*(normalized_throughput_channel_total_values_max[buffer_discipline_index, :] - normalized_throughput_channel_total_values_average[buffer_discipline_index, :])]
#     # ax1.errorbar(par_deadline_fraction_values, 100 * normalized_throughput_channel_total_values_average[buffer_discipline_index, :], yerr=yerr_total_1, color=colors[0], fmt='none')


plt.show()
