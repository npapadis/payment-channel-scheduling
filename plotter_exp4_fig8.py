# Plot results of buffering_capability experiments

from pypet import load_trajectory, pypetconstants
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

from save_legend import save_legend

save_at_directory = "./figures/"
Path(save_at_directory).mkdir(parents=True, exist_ok=True)
filename = 'results_121'


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
success_rate_channel_total_values = list(traj.f_get_from_runs('success_rate_channel_total', fast_access=True).values())
success_rate_channel_total_values = np.reshape(np.array(success_rate_channel_total_values), (len(par_scheduling_policy_values), len(par_buffer_discipline_values),  len(par_buffering_capability_values), len(par_max_buffering_time_values), len(par_seed_values)))
success_rate_channel_total_values_average = success_rate_channel_total_values.mean(axis=4)
success_rate_channel_total_values_max = success_rate_channel_total_values.max(axis=4)
success_rate_channel_total_values_min = success_rate_channel_total_values.min(axis=4)
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


plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"errorbar.capsize": 3})


for buffer_discipline_index, buffer_discipline in enumerate(par_buffer_discipline_values):
    for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values):
        fig, ax1 = plt.subplots()

        # Separate graph per policy
        # Success rate graph
        # yerr_total = [100*(success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - success_rate_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index]), 100*(success_rate_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index])]
        # ax1.bar(par_buffering_capability_values, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], yerr=yerr_total, label=scheduling_policy, color=colors[scheduling_policy_index])
        # Normalized throughput graph
        # yerr_total = [100*(normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - normalized_throughput_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index]), 100*(normalized_throughput_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index])]
        # ax1.bar(par_buffering_capability_values, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], yerr=yerr_total, label=scheduling_policy, color=colors[scheduling_policy_index], width=0.5)

        # One graph for all policies
        number_of_adjacent_bars = len(par_scheduling_policy_values)
        number_of_xticks = len(par_buffering_capability_values)
        bar_width = 1 / (number_of_adjacent_bars + 1)
        for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
            # Success rate graph
            yerr_total = [100*(success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - success_rate_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index]), 100*(success_rate_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index])]
            ax1.bar(np.arange(number_of_xticks) - number_of_adjacent_bars * bar_width / 2 + scheduling_policy_index * bar_width, 100 * success_rate_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], yerr=yerr_total, label=scheduling_policy, color=colors[scheduling_policy_index], width=bar_width)

            # Normalized throughput graph
            # yerr_total = [100*(normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - normalized_throughput_channel_total_values_min[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index]), 100*(normalized_throughput_channel_total_values_max[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index] - normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index])]
            # ax1.bar(np.arange(number_of_xticks) - number_of_adjacent_bars * bar_width / 2 + scheduling_policy_index * bar_width, 100 * normalized_throughput_channel_total_values_average[scheduling_policy_index, buffer_discipline_index, :, max_buffering_time_index], yerr=yerr_total, label=scheduling_policy, color=colors[scheduling_policy_index], width=bar_width)

        ax1.grid(True)
        ax1.set_axisbelow(True)
        ax1.set_ylim(bottom=0, top=100)
        ax1.set_xlabel("Buffering capability")
        ax1.set_ylabel("Success rate (%)")
        # ax1.set_ylabel("Normalized throughput (%)")

        ax1.set_xticks(np.linspace(0 - (number_of_adjacent_bars - 1) * bar_width / 4, number_of_xticks - (number_of_adjacent_bars - 1) * bar_width / 4, number_of_xticks + 1))
        ax1.set_xticklabels(par_buffering_capability_values, rotation=45)

        lines, labels = ax1.get_legend_handles_labels()
        legend = plt.legend(lines, labels, loc='best')
        fig.savefig(save_at_directory + filename + "_sr.png", bbox_inches='tight')
        fig.savefig(save_at_directory + filename + "_sr.pdf", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_nthr.png", bbox_inches='tight')
        # fig.savefig(save_at_directory + filename + "_nthr.pdf", bbox_inches='tight')

        legend_filename = filename + "_legend.png"
        save_legend(fig, lines, labels, legend, save_at_directory, legend_filename)
