from pypet import load_trajectory, pypetconstants, utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from cycler import cycler

# class Experiment_anatomy:
#     def __init__(self, who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time, seed, succeeded_per_amount, total_per_amount):
#         self.who_has_buffer = who_has_buffer
#         self.immediate_processing = immediate_processing
#         self.scheduling_policy = scheduling_policy
#         self.max_buffering_time = max_buffering_time
#         self.seed = seed
#         self.succeeded_per_amount = succeeded_per_amount
#         self.total_per_amount = total_per_amount


def save_legend(fig, lines, labels, legend, legend_fullpath):
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
    legend_fig.savefig(
        legend_fullpath,
        bbox_inches='tight',
        bbox_extra_artists=[legend_squared],
    )


# traj = load_trajectory(filename='./HDF5/results_full_2_rep.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)
traj = load_trajectory(filename='./HDF5/results.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/results_constant_amount_constant_deadline.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/results_constant_amount_constant_deadline_asymmetric_rates.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/results_constant_amount_50_constant_deadline_symmetric_rates.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)
# traj = load_trajectory(filename='./HDF5/results_constant_amount_50_constant_deadline_symmetric_rates_5_experiments.hdf5', name='single_channel_buffering', load_all=pypetconstants.LOAD_DATA)


# Parse parameter values
par_who_has_buffer_values = traj.f_get('who_has_buffer').f_get_range()
par_who_has_buffer_values = list(dict.fromkeys(par_who_has_buffer_values))
par_immediate_processing_values = traj.f_get('immediate_processing').f_get_range()
par_immediate_processing_values = list(dict.fromkeys(par_immediate_processing_values))
par_scheduling_policy_values = traj.f_get('scheduling_policy').f_get_range()
par_scheduling_policy_values = list(dict.fromkeys(par_scheduling_policy_values))
par_max_buffering_time_values = traj.f_get('max_buffering_time').f_get_range()
par_max_buffering_time_values = list(dict.fromkeys(par_max_buffering_time_values))
par_seed_values = traj.f_get('seed').f_get_range()
par_seed_values = list(dict.fromkeys(par_seed_values))
# Bad alternatives:
# xs = list(set(xs)) # is in random order
# xs = utils.explore.find_unique_points([traj.f_get('who_has_buffer')]) # needs further processing and uses an ordered dict already


# success_rates_0_values = list(traj.f_get_from_runs('success_rate_0', fast_access=True).values())
# success_rates_0_values = np.reshape(np.array(success_rates_0_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_rates_0_values_average = success_rates_0_values.mean(axis=4)
# success_rates_1_values = list(traj.f_get_from_runs('success_rate_1', fast_access=True).values())
# success_rates_1_values = np.reshape(np.array(success_rates_1_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# success_rates_1_values_average = success_rates_1_values.mean(axis=4)
success_rates_total_values = list(traj.f_get_from_runs('success_rate_total', fast_access=True).values())
success_rates_total_values = np.reshape(np.array(success_rates_total_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
success_rates_total_values_average = success_rates_total_values.mean(axis=4)

# throughput_0_values = list(traj.f_get_from_runs('throughput_0', fast_access=True).values())
# throughput_0_values = np.reshape(np.array(throughput_0_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# throughput_0_values_average = throughput_0_values.mean(axis=4)
# throughput_1_values = list(traj.f_get_from_runs('throughput_1', fast_access=True).values())
# throughput_1_values = np.reshape(np.array(throughput_1_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# throughput_1_values_average = throughput_1_values.mean(axis=4)
#
# sacrificed_0_values = list(traj.f_get_from_runs('sacrificed_0', fast_access=True).values())
# sacrificed_0_values = np.reshape(np.array(sacrificed_0_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_0_values_average = sacrificed_0_values.mean(axis=4)
# sacrificed_1_values = list(traj.f_get_from_runs('sacrificed_1', fast_access=True).values())
# sacrificed_1_values = np.reshape(np.array(sacrificed_1_values), (len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# sacrificed_1_values_average = sacrificed_1_values.mean(axis=4)


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmky')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = cycler(color='bgrcmky')
markers = [".", "x", "s", "+", "*", "d"]


# # Plot success_rate vs max buffering time for all who_has_buffer
# for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
#     for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#         fig, ax1 = plt.subplots()
#         # ax1.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
#         ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
#         ax3 = ax1.twinx()  # instantiate a third axis that shares the same x-axis
#         # ax2.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
#         # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'black']
#         for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
#             # linestyle = linestyles[who_has_buffer_index]
#             innermost_index = who_has_buffer_index
#             color = colors[innermost_index]
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Buffers: "+who_has_buffer, linestyle='solid')
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Buffers: "+who_has_buffer, linestyle='solid')
#             ax1.plot(par_max_buffering_time_values, 100 * success_rates_total_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate total, Buffers: "+who_has_buffer, linestyle=linestyles[0], color=color)
#             # ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 0, Buffers: "+who_has_buffer, linestyle='dashed')
#             # ax2.plot(par_max_buffering_time_values, throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 1, Buffers: "+who_has_buffer, linestyle='dashed')
#             ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Total throughput, Buffers: "+who_has_buffer, linestyle=linestyles[1], color=color)
#             ax3.plot(par_max_buffering_time_values, sacrificed_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + sacrificed_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Number of sacrificed transactions, Buffers: "+who_has_buffer, linestyle=linestyles[3], color=color, alpha=1)
#         # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_1_values_average], label='Success rate of node 1')
#         # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_total_values_average], label='Overall success rate')
#
#         ax1.set_ylim(bottom=0, top=100)
#         ax2.set_ylim(bottom=0)
#         ax3.set_ylim(bottom=0)
#         ax1.set_xlabel("Maximum buffering time (sec)")
#         ax1.set_ylabel("Success rate (%)")
#         ax2.set_ylabel("Successful amount (coins)")
#         ax2.tick_params(axis='y', rotation=45)
#         ax3.set_ylabel("Number of sacrificed transactions")
#         ax3.spines["right"].set_position(("axes", 1.2))
#         plt.title("Success rate as a function of maximum buffering time for a single channel\nImmediate processing: {}, Processing order: {}".format(immediate_processing, scheduling_policy))
#         fig.savefig("./figures/sr_wrt_mbt/1. sr_wrt_mbt_ap_whb/fig1-_-"+str(immediate_processing_index+1)+"-"+str(scheduling_policy_index+1)+".png", bbox_inches='tight')
#
#         lines_1, labels_1 = ax1.get_legend_handles_labels()
#         lines_2, labels_2 = ax2.get_legend_handles_labels()
#         lines_3, labels_3 = ax3.get_legend_handles_labels()
#         lines = lines_1 + lines_2 + lines_3
#         labels = labels_1 + labels_2 + labels_3
#         legend = plt.legend(lines, labels)
#         legend_fullpath = "./figures/sr_wrt_mbt/1. sr_wrt_mbt_ap_whb/legend1-_-A-A.png"
#         save_legend(fig, lines, labels, legend, legend_fullpath)
#
#         # legd = ax1.legend(lines, labels, loc='best')
#         # legd = fig.legend(lines, labels, loc='center right')
#         # legend = plt.legend(lines, labels)
#         # legend.savefig("legend1-_-*-*")
# # plt.show()
#
#
# # Plot success_rate vs max buffering time for all immediate processing
# for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
#     for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#         fig, ax1 = plt.subplots()
#         # ax1.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
#         ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
#         ax3 = ax1.twinx()  # instantiate a third axis that shares the same x-axis
#         # ax2.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
#         # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'black']
#         for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
#             innermost_index = immediate_processing_index
#             color = colors[innermost_index]
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Immediate Processing: "+immediate_processing, linestyle='solid')
#             # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Immediate Processing: "+immediate_processing, linestyle='solid')
#             ax1.plot(par_max_buffering_time_values, 100 * success_rates_total_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate total, Immediate Processing: "+str(immediate_processing), linestyle=linestyles[0], color=color)
#             # ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 0, Immediate Processing: "+str(immediate_processing), linestyle='dashed')
#             # ax2.plot(par_max_buffering_time_values, throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 1, Immediate Processing: "+str(immediate_processing), linestyle='dashed')
#             ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Total throughput, Immediate Processing: "+str(immediate_processing), linestyle=linestyles[1], color=color)
#             ax3.plot(par_max_buffering_time_values, sacrificed_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + sacrificed_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Number of sacrificed transactions, Immediate Processing: "+str(immediate_processing), linestyle=linestyles[3], color=color, alpha=1)
#         # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_1_values_average], label='Success rate of node 1')
#         # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_total_values_average], label='Overall success rate')
#
#         ax1.set_ylim(bottom=0, top=100)
#         ax2.set_ylim(bottom=0)
#         ax3.set_ylim(bottom=0)
#         ax1.set_xlabel("Maximum buffering time (sec)")
#         ax1.set_ylabel("Success rate (%)")
#         ax2.set_ylabel("Successful amount (coins)")
#         ax2.tick_params(axis='y', rotation=45)
#         ax3.set_ylabel("Number of sacrificed transactions")
#         ax3.spines["right"].set_position(("axes", 1.2))
#         plt.title("Success rate as a function of maximum buffering time for a single channel\nWho has buffer: {}, Processing order: {}".format(who_has_buffer, scheduling_policy))
#         fig.savefig("./figures/sr_wrt_mbt/2. sr_wrt_mbt_ap_ip/fig1-"+str(who_has_buffer_index+1)+"-_-"+str(scheduling_policy_index+1)+".png", bbox_inches='tight')
#
#         lines_1, labels_1 = ax1.get_legend_handles_labels()
#         lines_2, labels_2 = ax2.get_legend_handles_labels()
#         lines_3, labels_3 = ax3.get_legend_handles_labels()
#         lines = lines_1 + lines_2 + lines_3
#         labels = labels_1 + labels_2 + labels_3
#         legend = plt.legend(lines, labels)
#         legend_fullpath = "./figures/sr_wrt_mbt/2. sr_wrt_mbt_ap_ip/legend1-A-_-A.png"
#         save_legend(fig, lines, labels, legend, legend_fullpath)
#
#         # fig.legend(lines, labels, loc='best')
#         # legend = plt.legend(lines, labels)
#         # legend.savefig("legend1-*-_-*")
# # plt.show()
#
#
# Plot success_rate vs max buffering time for all processing orders
for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
    for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
        fig, ax1 = plt.subplots()
        # ax1.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
        # ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        # ax3 = ax1.twinx()  # instantiate a third axis that shares the same x-axis
        # ax2.set_prop_cycle(color=['red', 'green', 'blue', 'orange', 'magenta'])
        # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'black']
        for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
            innermost_index = scheduling_policy_index
            color = colors[innermost_index]
            # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Processing order: "+scheduling_policy, linestyle='solid')
            # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Processing order: "+scheduling_policy, linestyle='solid')
            ax1.plot(par_max_buffering_time_values, 100 * success_rates_total_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate total, Processing order: "+scheduling_policy, linestyle=linestyles[0], color=color, alpha=1)
            # # ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 0, Processing order: "+scheduling_policy, linestyle='dashed')
            # # ax2.plot(par_max_buffering_time_values, throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Throughput of node 1, Processing order: "+scheduling_policy, linestyle='dashed')
            # ax2.plot(par_max_buffering_time_values, throughput_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + throughput_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Total throughput, Processing order: "+scheduling_policy, linestyle=linestyles[1], color=color, alpha=1)
            # ax3.plot(par_max_buffering_time_values, sacrificed_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :] + sacrificed_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Number of sacrificed transactions, Processing order: "+scheduling_policy, linestyle=linestyles[3], color=color, alpha=1)
        # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_1_values_average], label='Success rate of node 1')
        # ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_total_values_average], label='Overall success rate')

        ax1.set_ylim(bottom=0, top=100)
        # ax2.set_ylim(bottom=0)
        # ax3.set_ylim(bottom=0)
        ax1.set_xlabel("Maximum buffering time (sec)")
        ax1.set_ylabel("Success rate (%)")
        # ax2.set_ylabel("Successful amount (coins)")
        # ax2.tick_params(axis='y', rotation=45)
        # ax3.set_ylabel("Number of sacrificed transactions")
        # ax3.spines["right"].set_position(("axes", 1.2))
        plt.title("Success rate as a function of maximum buffering time for a single channel\nWho has buffer: {}, Immediate processing: {}".format(who_has_buffer, immediate_processing))
        fig.savefig("./figures/sr_wrt_mbt/3. sr_wrt_mbt_ap_pror/fig1-"+str(who_has_buffer_index+1)+"-"+str(immediate_processing_index+1)+"-_.png", bbox_inches='tight')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # lines_3, labels_3 = ax3.get_legend_handles_labels()
        # lines = lines_1 + lines_2 + lines_3
        # labels = labels_1 + labels_2 + labels_3
        lines = lines_1
        labels = labels_1
        legend = plt.legend(lines, labels)
        legend_fullpath = "./figures/sr_wrt_mbt/3. sr_wrt_mbt_ap_pror/legend1-A-A-_.png"
        save_legend(fig, lines, labels, legend, legend_fullpath)

        # handles, labels = ax1.get_legend_handles_labels()
        # axe.legend(handles, labels, loc=loc)
        # axe.xaxis.set_visible(False)
        # axe.yaxis.set_visible(False)
        # for v in axe.spines.values():
        #     v.set_visible(False)

        # fig_leg = plt.figure()
        # ax_leg = fig_leg.add_subplot(111)
        # # add the legend from the previous axes
        # ax_leg.legend(*ax1.get_legend_handles_labels(), loc='center')
        # # hide the axes frame and the x/y labels
        # ax_leg.axis('off')
        # fig_leg.savefig('./figures/legend.png')



        # legend = plt.legend(lines, labels)
        # legend.savefig("legend1-*-*-_")
# plt.show()





exit()


# =========================================================================================









all_transactions_list_from_runs = traj.f_get_from_runs('all_transactions_list', fast_access=True)
# success_rate_per_amount = np.array((len(par_who_has_buffer_values), len(par_immediate_processing_values), len(par_scheduling_policy_values), len(par_max_buffering_time_values), len(par_seed_values)))
# experiment_anatomy_list = []
experiment_anatomy_dict = {}
experiment_anatomy_summary_dict = {}

for index in all_transactions_list_from_runs:
    traj.v_idx = index
    succeeded_per_amount = {}
    total_per_amount = {}
    success_rate_per_amount = {}
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
            success_rate_per_amount[amount] = (succeeded_per_amount[amount]) / (total_per_amount[amount])
        else:
            success_rate_per_amount[amount] = 0

    # print(succeeded_per_amount)
    # print(total_per_amount)
    # experiment_anatomy_list.append(Experiment_anatomy(traj.who_has_buffer, traj.immediate_processing, traj.scheduling_policy, traj.max_buffering_time, traj.seed, succeeded_per_amount, total_per_amount))
    # experiment_anatomy_dict.setdefault((traj.who_has_buffer, traj.immediate_processing, traj.scheduling_policy, traj.max_buffering_time, traj.seed), []).append(success_rate_per_amount)
    experiment_anatomy_dict[(traj.who_has_buffer, traj.immediate_processing, traj.scheduling_policy, traj.max_buffering_time, traj.seed)] = success_rate_per_amount
    # experiment_anatomy_summary_dict((traj.who_has_buffer, traj.immediate_processing, traj.scheduling_policy, traj.max_buffering_time))


# for (who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time) in (par_who_has_buffer_values, par_immediate_processing_values, par_scheduling_policy_values, par_max_buffering_time_values):
for immediate_processing in par_immediate_processing_values:
    for scheduling_policy in par_scheduling_policy_values:
        for who_has_buffer in par_who_has_buffer_values:
            for max_buffering_time in par_max_buffering_time_values:
                group_index = (who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)
                experiment_anatomy_summary_dict[group_index] = {}
                for seed in par_seed_values:
                    experiment_index = (who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time, seed)
                    for amount in experiment_anatomy_dict[experiment_index].keys():
                        if amount in experiment_anatomy_summary_dict[group_index].keys():
                            experiment_anatomy_summary_dict[group_index][amount] += experiment_anatomy_dict[experiment_index][amount]
                        else:
                            experiment_anatomy_summary_dict[group_index][amount] = experiment_anatomy_dict[experiment_index][amount]
                    for amount in experiment_anatomy_summary_dict[group_index].keys():
                        experiment_anatomy_summary_dict[group_index][amount] /= len(par_seed_values)
                    # Sort by amount
                    experiment_anatomy_summary_dict[group_index] = dict(sorted(experiment_anatomy_summary_dict[group_index].items()))

    # index_whb = par_who_has_buffer_values.index(traj.who_has_buffer)
    # index_ip = par_immediate_processing_values.index(traj.immediate_processing)
    # index_pror = par_scheduling_policy_values.index(traj.scheduling_policy)
    # index_mbt = par_max_buffering_time_values.index(traj.max_buffering_time)
    # index_seed = par_seed_values.index(traj.seed)
    # success_rate_per_amount


mbt_step = min(len(par_max_buffering_time_values), max(4, (len(par_max_buffering_time_values)//4)))
bin_count = 10

# Plot success_rate vs transaction amount for all who_has_buffer
for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
    for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
        for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values[::mbt_step]):
            fig, ax1 = plt.subplots()
            for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
                innermost_index = who_has_buffer_index
                color = colors[innermost_index]

                amounts_to_process = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].keys())
                success_rates_to_process = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].values())
                # bin_width = max(np.floor(max(amounts_to_process)) // bin_count)
                # amounts_to_plot, bin_edges = np.histogram(amounts_to_process, bins=bin_count, weights=success_rates_to_process)
                elements_per_bin, bin_edges = np.histogram(amounts_to_process, bins=bin_count)
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
                #     weights[amount_index] = success_rates_to_process[amount_index] / elements_per_bin[bin_index]

                bin_per_tx = np.digitize(amounts_to_process, bin_edges)
                weights = [success_rates_to_process[i] / bin_per_tx[b] for i, b in enumerate(bin_per_tx)]
                ax1.hist(amounts_to_process, weights=weights, bins=bin_count, label="Success rate, Who has buffer: "+who_has_buffer, color=color)

                # ax1.hist(amounts_to_process, weights=weights, bins=bin_count, label="Success rate, Who has buffer: "+who_has_buffer, color=color)
                # plt.hist(amounts_to_process, weights=weights, bins=bin_count, label="Success rate, Who has buffer: "+who_has_buffer, color=color)

                # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Buffers: "+who_has_buffer, linestyle='solid')
                # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Buffers: "+who_has_buffer, linestyle='solid')
                # ax1.plot(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+who_has_buffer, linestyle=linestyles[0], color=color)
                # ax1.bar(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+who_has_buffer, linestyle=linestyles[0], color=color)
                # ax1.scatter(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Who has buffer: "+who_has_buffer, marker=markers[innermost_index], color=color)

            ax1.set_ylim(bottom=0, top=100)
            ax1.set_xlabel("Transaction amount (coins)")
            ax1.set_ylabel("Success rate (%)")
            plt.title("Success rate as a function of transaction amount for a single channel\nImmediate processing: {}, Processing order: {}, Max buffering time: {}".format(immediate_processing, scheduling_policy, max_buffering_time))
            fig.savefig("./figures/sr_wrt_ta/1. sr_wrt_ta_ap_whb/fig2-_-"+str(immediate_processing_index+1)+"-"+str(scheduling_policy_index+1)+"-"+str(max_buffering_time_index+1)+".png", bbox_inches='tight')

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines = lines_1
            labels = labels_1
            legend = plt.legend(lines, labels)
            legend_fullpath = "./figures/sr_wrt_ta/1. sr_wrt_ta_ap_whb/legend2-_-A-A-A.png"
            save_legend(fig, lines, labels, legend, legend_fullpath)
# plt.show()

#
# # Plot success_rate vs transaction amount for all immediate_processing
# for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
#     for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#         for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values[::mbt_step]):
#             fig, ax1 = plt.subplots()
#             for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
#                 innermost_index = immediate_processing_index
#                 color = colors[innermost_index]
#                 amounts_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].keys())
#                 success_rates_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].values())
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Immediate Processing: "+str(immediate_processing), linestyle=linestyles[0], color=color)
#                 # ax1.bar(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Immediate Processing: "+str(immediate_processing), linestyle=linestyles[0], color=color)
#                 ax1.scatter(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Immediate Processing: "+str(immediate_processing), marker=markers[innermost_index], color=color)
#
#             ax1.set_ylim(bottom=0, top=100)
#             ax1.set_xlabel("Transaction amount (coins)")
#             ax1.set_ylabel("Success rate (%)")
#             plt.title("Success rate as a function of transaction amount for a single channel\nWho has buffer: {}, Processing order: {}, Max buffering time: {}".format(who_has_buffer, scheduling_policy, max_buffering_time))
#             fig.savefig("./figures/sr_wrt_ta/2. sr_wrt_ta_ap_ip/fig2-"+str(who_has_buffer_index+1)+"-_-"+str(scheduling_policy_index+1)+"-"+str(max_buffering_time_index+1)+".png", bbox_inches='tight')
#
#             lines_1, labels_1 = ax1.get_legend_handles_labels()
#             lines = lines_1
#             labels = labels_1
#             legend = plt.legend(lines, labels)
#             legend_fullpath = "./figures/sr_wrt_ta/2. sr_wrt_ta_ap_ip/legend2-A-_-A-A.png"
#             save_legend(fig, lines, labels, legend, legend_fullpath)
# # plt.show()
#
# # Plot success_rate vs transaction amount for all scheduling_policy
# for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
#     for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
#         for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values[::mbt_step]):
#             fig, ax1 = plt.subplots()
#             for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#                 innermost_index = scheduling_policy_index
#                 color = colors[innermost_index]
#                 amounts_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].keys())
#                 success_rates_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].values())
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Processing order: "+scheduling_policy, linestyle=linestyles[0], color=color)
#                 # ax1.bar(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Processing order: "+scheduling_policy, linestyle=linestyles[0], color=color)
#                 ax1.scatter(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Processing order: "+scheduling_policy, marker=markers[innermost_index], color=color)
#
#             ax1.set_ylim(bottom=0, top=100)
#             ax1.set_xlabel("Transaction amount (coins)")
#             ax1.set_ylabel("Success rate (%)")
#             plt.title("Success rate as a function of transaction amount for a single channel\nWho has buffer: {}, Immediate processing: {}, Max buffering time: {}".format(who_has_buffer, immediate_processing, max_buffering_time))
#             fig.savefig("./figures/sr_wrt_ta/3. sr_wrt_ta_ap_pror/fig2-"+str(who_has_buffer_index+1)+"-"+str(immediate_processing_index+1)+"-_-"+str(max_buffering_time_index+1)+".png", bbox_inches='tight')
#
#             lines_1, labels_1 = ax1.get_legend_handles_labels()
#             lines = lines_1
#             labels = labels_1
#             legend = plt.legend(lines, labels)
#             legend_fullpath = "./figures/sr_wrt_ta/3. sr_wrt_ta_ap_pror/legend2-A-A-_-A.png"
#             save_legend(fig, lines, labels, legend, legend_fullpath)
# # plt.show()
#
# # Plot success_rate vs transaction amount for all max_buffering_time
# for who_has_buffer_index, who_has_buffer in enumerate(par_who_has_buffer_values):
#     for immediate_processing_index, immediate_processing in enumerate(par_immediate_processing_values):
#         for scheduling_policy_index, scheduling_policy in enumerate(par_scheduling_policy_values):
#             fig, ax1 = plt.subplots()
#             for max_buffering_time_index, max_buffering_time in enumerate(par_max_buffering_time_values[::mbt_step]):
#                 innermost_index = max_buffering_time_index
#                 color = colors[innermost_index]
#                 amounts_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].keys())
#                 success_rates_to_plot = list(experiment_anatomy_summary_dict[(who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time)].values())
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_0_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 0, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(par_max_buffering_time_values, 100 * success_rates_1_values_average[who_has_buffer_index, immediate_processing_index, scheduling_policy_index, :], label="Success rate of node 1, Buffers: "+who_has_buffer, linestyle='solid')
#                 # ax1.plot(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Max buffering time: "+str(max_buffering_time), linestyle=linestyles[0], color=color)
#                 # ax1.bar(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Max buffering time: "+str(max_buffering_time), linestyle=linestyles[0], color=color)
#                 ax1.scatter(amounts_to_plot, [100 * elem for elem in success_rates_to_plot], label="Success rate, Max buffering time: "+str(max_buffering_time), marker=markers[innermost_index], color=color)
#
#             ax1.set_ylim(bottom=0, top=100)
#             ax1.set_xlabel("Transaction amount (coins)")
#             ax1.set_ylabel("Success rate (%)")
#             plt.title("Success rate as a function of transaction amount for a single channel\nWho has buffer: {}, Immediate processing: {}, Processing order: {}".format(who_has_buffer, immediate_processing, scheduling_policy))
#             fig.savefig("./figures/sr_wrt_ta/4. sr_wrt_ta_ap_mbt/fig2-"+str(who_has_buffer_index+1)+"-"+str(immediate_processing_index+1)+"-"+str(scheduling_policy_index+1)+"_.png", bbox_inches='tight')
#
#             lines_1, labels_1 = ax1.get_legend_handles_labels()
#             lines = lines_1
#             labels = labels_1
#             legend = plt.legend(lines, labels)
#             legend_fullpath = "./figures/sr_wrt_ta/4. sr_wrt_ta_ap_mbt/legend2-A-A-A-_.png"
#             save_legend(fig, lines, labels, legend, legend_fullpath)
# # plt.show()






























#======================================================================================
# print("------------")


# for tau_ref, I_col in traj.iteritems():
#     plt.plot(I_col.index, I_col, label='Avg. Rate for tau_ref=%s' % str(tau_ref))

# for run_name in traj.f_get_run_names():
#     traj.f_set_crun(run_name)
#     success_rates_0_values_average = traj.crun.success_rate_0
#     success_rates_1_values_average = traj.crun.success_rate_1
#     print('%s: success_rates_0_values_average=%f, success_rates_1_values_average=%f' % (run_name, success_rates_0_values_average, success_rates_1_values_average))
# traj.f_restore_default()

# mydict = traj.f_get_from_runs('success_rate_0', fast_access=False)
# srs = mydict.values()
# --- or ---
# srs = traj.f_get_from_runs('success_rate_0', fast_access=False).values()
# traj.f_load_items(srs)
# srs_values = [x.f_get() for x in srs]

# srs_values = list(traj.f_get_from_runs('success_rate_0', fast_access=True).values())
# print(srs_values)
# helpe = traj.f_get_explored_parameters(fast_access=True)



# results_dataframe = pd.DataFrame(columns=['who_has_buffer', 'immediate_processing', 'scheduling_policy', 'max_buffering_time', 'seed'])
# print(results_dataframe)
# This frame is basically a two dimensional table that we can index with our
# parameters

# results_dataframe = pd.read_hdf('./HDF5/results.hdf5')
# print(results_dataframe)

# Now iterate over the results. The result list is a list of tuples, with the
# run index at first position and our result at the second
# for result_tuple in result_list:
#     run_idx = result_tuple[0]
#     firing_rates = result_tuple[1]
#     I_val = I_range[run_idx]
#     ref_val = ref_range[run_idx]
#     rates_frame.loc[I_val, ref_val] = firing_rates  # Put the firing rate into the
#     # data frame
#
# df_gp_1 = results_dataframe[['User_ID', 'Purchase']].groupby('User_ID').agg(np.mean).reset_index()




# # Finally we going to store our new firing rate table into the trajectory
# traj.f_add_result('summary.firing_rates', rates_frame=rates_frame,
#                   comment='Contains a pandas data frame with all firing rates.')





# # And now we want to find som particular results, the ones where x was 2 or y was 8.
# # Therefore, we use a lambda function
# my_filter_predicate= lambda x,y: x==2 or y==8
#
# # We can now use this lambda function to search for the run indexes associated with x==2 OR y==8.
# # We need a list specifying the names of the parameters and the predicate to do this.
# # Note that names need to be in the order as listed in the lambda function, here 'x' and 'y':
# idx_iterator = traj.f_find_idx(['x','y'], my_filter_predicate)
#
# # Now we can print the corresponding results:
# print('The run names and results for parameter combinations with x==2 or y==8:')
# for idx in idx_iterator:
#     # We focus on one particular run. This is equivalent to calling `traj.f_set_crun(idx)`.
#     traj.v_idx=idx
#     run_name = traj.v_crun
#     # and print everything nicely
#     print('%s: x=%d, y=%d, z=%d' %(run_name, traj.x, traj.y, traj.crun.z))
#




# Plot success_rate vs who_has_buffer for all combinations
# fig, ax = plt.subplots()
# idx_iterator = traj.f_find_idx(['who_has_buffer', 'immediate_processing', 'scheduling_policy', 'max_buffering_time', 'seed'],
#                                (lambda who_has_buffer, immediate_processing, scheduling_policy, max_buffering_time, seed: (immediate_processing is True) and (scheduling_policy == "oldest_transaction_first") and (seed == 87563))
#                                )
# mbt_plot = []
# sr0_plot = []
# for idx in idx_iterator:
#     # We focus on one particular run. This is equivalent to calling `traj.f_set_crun(idx)`.
#     traj.v_idx = idx
#     run_name = traj.v_crun
#     mbt_plot.append(traj.max_buffering_time)
#     sr0_plot.append(traj.crun.success_rate_0)
# ax.plot(mbt_plot, sr0_plot)
# legend = ax.legend(loc='best')
# plt.show()
# print(mbt_plot)
# print(sr0_plot)
#
# #     print('%s: who_has_buffer=%s, max_buffering_time=%d, seed=%d, success_rate_total=%d' % (run_name, traj.who_has_buffer, traj.max_buffering_time, traj.seed, 100*traj.crun.success_rate_total))
# # traj.f_restore_default()




