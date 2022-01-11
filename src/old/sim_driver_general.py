import matplotlib.pyplot as plt
import numpy as np

# from DES.sc_DES_no_buffer_fun import sc_DES_no_buffer_fun
# from DES.sc_DES_with_buffer_fun import sc_DES_with_buffer_fun
from DES.sc_DES_with_all_kinds_of_buffers_fun import sc_DES_with_all_kinds_of_buffers_fun

np.set_printoptions(precision=2)

# SIMULATION PARAMETERS

initial_balances = [150, 150]
# total_simulation_time = 1000.0

# Node 0:
total_transactions_0 = 100
max_transaction_amount_0 = sum(initial_balances)
exp_mean_0 = 1 / 3

# Node 1:
total_transactions_1 = 100
max_transaction_amount_1 = sum(initial_balances)
exp_mean_1 = 1 / 3

# This variable can take values "none", "only_node_0", "only_node_1", "both_separate", "both_shared". Any other value results in an error.
who_has_buffer_values = ["both_shared"]#["none", "only_node_0", "only_node_1", "both_separate", "both_shared"]
immediate_processing = False
# This variable can take values "oldest_transaction_first" (FIFO), "youngest_transaction_first" (LIFO), "closest_deadline_first", "largest_amount_first", "smallest_amount_first". Any other value results in an error.
processing_order_values = ["oldest_transaction_first"]#, "youngest_transaction_first", "closest_deadline_first", "largest_amount_first", "smallest_amount_first"]
verbose = False
# verbose = True

# lambda_total = 1/15
# total_transactions = 1000
max_buffering_time_values = range(0, 100, 50)
num_of_experiments = 3    # number of num_of_experiments for every experiment


# Plot success rates for each node separately for different values of delays

success_rates_0_values_all = np.ndarray([len(who_has_buffer_values), len(processing_order_values), len(max_buffering_time_values), num_of_experiments])
success_rates_1_values_all = np.ndarray([len(who_has_buffer_values), len(processing_order_values), len(max_buffering_time_values), num_of_experiments])
success_rates_total_values_all = np.ndarray([len(who_has_buffer_values), len(processing_order_values), len(max_buffering_time_values), num_of_experiments])
sacrificed_0 = np.ndarray([len(who_has_buffer_values), len(processing_order_values), len(max_buffering_time_values), num_of_experiments])
sacrificed_1 = np.ndarray([len(who_has_buffer_values), len(processing_order_values), len(max_buffering_time_values), num_of_experiments])
# balance_history_node_0 = np.ndarray([len(max_allowed_delay_values), num_of_experiments])

experiment_count = 0
for who_has_buffer_index, who_has_buffer in enumerate(who_has_buffer_values):
    for processing_order_index, processing_order in enumerate(processing_order_values):
        for max_buffering_time_index, max_buffering_time in enumerate(max_buffering_time_values):
            experiment_count += 1
            print("\nStarting experiment #{} out of {}.".format(experiment_count, len(max_buffering_time_values) * len(processing_order_values) * len(who_has_buffer_values)))
            for r in range(num_of_experiments):
                print("\tStarting repetition #{} out of {}.".format(r + 1, num_of_experiments))
                seed = 12345 + r
                sr, sa, bh0, sac = sc_DES_with_all_kinds_of_buffers_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_buffering_time, who_has_buffer, immediate_processing, processing_order, verbose, seed)
                # sr, sa, bh0, atl = sc_DES_with_all_kinds_of_buffers_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_buffering_time, who_has_buffer, immediate_processing, processing_order, verbose, seed)
                success_rates_0_values_all[who_has_buffer_index, processing_order_index, max_buffering_time_index, r] = sr[0]
                success_rates_1_values_all[who_has_buffer_index, processing_order_index, max_buffering_time_index, r] = sr[1]
                success_rates_total_values_all[who_has_buffer_index, processing_order_index, max_buffering_time_index, r] = sr[2]
                sacrificed_0[who_has_buffer_index, processing_order_index, max_buffering_time_index, r] = int(sac[0])
                sacrificed_1[who_has_buffer_index, processing_order_index, max_buffering_time_index, r] = int(sac[1])
                # all_transactions_list_node_0, all_transactions_list_node_1 = atl[0], atl[1]
                # balance_history_node_0[who_has_buffer_index, processing_order_index, delay_index] = bh0

success_rates_0_values_average = success_rates_0_values_all.mean(axis=3)
success_rates_1_values_average = success_rates_1_values_all.mean(axis=3)
success_rates_total_values_average = success_rates_total_values_all.mean(axis=3)
sacrificed_0_average = sacrificed_0.mean(axis=3)
sacrificed_1_average = sacrificed_1.mean(axis=3)

print("\nsuccess_rates_0_values_average = ", success_rates_0_values_average)
print("\nsuccess_rates_1_values_average = ", success_rates_1_values_average)
print("\nsuccess_rates_total_values_average = ", success_rates_total_values_average)
print(sacrificed_0_average)
print(sacrificed_1_average)
# print(balance_history_node_0)
# print(all_transactions_list_node_0)
# print(all_transactions_list_node_1)

# fig, ax = plt.subplots()
# ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_0_values_average], label='Success rate of node 0')
# ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_1_values_average], label='Success rate of node 1')
# ax.plot(max_buffering_time_values, [100 * elem for elem in success_rates_total_values_average], label='Overall success rate')
# ax.set_ylim([0, 100])
# legend = ax.legend(loc='best')
# plt.xlabel("Maximum allowed delay (sec)")
# plt.ylabel("Success rate (%)")
# plt.title("Success rate as a function of maximum allowed delay for a single channel")
# plt.show()

# # Plot balance history of node 0 from a single simulation
# plt.figure(2)
# plt.plot(balance_history_node_0[0][0][0], balance_history_node_0[0][0][1])
# plt.xlabel("Simulation time")
# plt.ylabel("Balance of node 0")
# plt.title("Balance of node 0 for a single experiment ")
# plt.show()
