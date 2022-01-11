import matplotlib.pyplot as plt
import numpy as np

# from DES.sc_DES_no_buffer_fun import sc_DES_no_buffer_fun
# from DES.sc_DES_with_buffer_fun import sc_DES_with_buffer_fun
from DES.sc_DES_with_all_kinds_of_buffers_fun import sc_DES_with_all_kinds_of_buffers_fun

# SIMULATION PARAMETERS

initial_balances = [150, 150]
# total_simulation_time = 1000.0

# Node 0:
total_transactions_0 = 200
max_transaction_amount_0 = sum(initial_balances)
exp_mean_0 = 1 / 3

# Node 1:
total_transactions_1 = 200
max_transaction_amount_1 = sum(initial_balances)
exp_mean_1 = 1 / 3

# This variable can take values "none", "only_node_0", "only_node_1", "both_separate", "both_shared". Any other value results in an error.
who_has_buffer_values = ["none", "only_node_0", "only_node_1", "both_separate", "both_shared"]
immediate_processing = True
# This variable can take values "oldest_transaction_first" (FIFO), "youngest_transaction_first" (LIFO), "closest_deadline_first", "largest_amount_first", "smallest_amount_first". Any other value results in an error.
processing_order_values = ["oldest_transaction_first", "youngest_transaction_first", "closest_deadline_first", "largest_amount_first", "smallest_amount_first"]
verbose = False
# verbose = True

# lambda_total = 1/15
# total_transactions = 1000
max_allowed_delay_values = range(0, 300, 50)
repetitions = 5    # number of num_of_experiments for every experiment


# Plot success rates for each node separately for different values of delays

success_rates_0_values_all = []       # matrix of size num_of_who_has_buffer_values x num_of_processing_order_values x num_of_delay_values x num_of_experiments
success_rates_1_values_all = []       # matrix of size num_of_who_has_buffer_values x num_of_processing_order_values x num_of_delay_values x num_of_experiments
success_rates_total_values_all = []   # matrix of size num_of_who_has_buffer_values x num_of_processing_order_values x num_of_delay_values x num_of_experiments
balance_history_node_0 = []           # matrix of size num_of_delay_values x num_of_experiments

for who_has_buffer_index in range(len(who_has_buffer_values)):
    who_has_buffer = who_has_buffer_values[who_has_buffer_index]
    success_rates_0_values_all.append([])
    success_rates_1_values_all.append([])
    success_rates_total_values_all.append([])
    balance_history_node_0.append([])
    for processing_order in processing_order_values:
        success_rates_0_values_all[who_has_buffer_index].append([])
        success_rates_1_values_all[who_has_buffer_index].append([])
        success_rates_total_values_all[who_has_buffer_index].append([])
        balance_history_node_0[who_has_buffer_index].append([])
        for index_delay in range(len(max_allowed_delay_values)):
            print("\nStarting experiment #{} out of {}.".format(index_delay + 1, len(max_allowed_delay_values)))
            max_allowed_delay = max_allowed_delay_values[index_delay]


            for r in range(repetitions):
                print("\tStarting repetition #{} out of {}.".format(r+1, repetitions))
                seed = 12345 + r
                # s = sc_DES_no_buffer_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay, verbose, seed)
                # sr, sa, bh0 = sc_DES_with_buffer_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay, verbose, seed)
                # sr, sa, bh0 = sc_DES_with_all_kinds_of_buffers_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay, who_has_buffer, immediate_processing, verbose, seed)
                sr, sa, bh0 = sc_DES_with_all_kinds_of_buffers_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay, who_has_buffer, immediate_processing, processing_order, verbose, seed)
                success_rates_0_values_all[index_delay].append(sr[0])
                success_rates_1_values_all[index_delay].append(sr[1])
                success_rates_total_values_all[index_delay].append(sr[2])
                balance_history_node_0[index_delay].append(bh0)

# print(success_rates_values_all)

success_rates_0_values_average = []
success_rates_1_values_average = []
success_rates_total_values_average = []
for index_delay in range(len(max_allowed_delay_values)):
    success_rates_0_values_average.append((1.0 * sum(success_rates_0_values_all[index_delay])) / len(success_rates_0_values_all[index_delay]))
    success_rates_1_values_average.append((1.0 * sum(success_rates_1_values_all[index_delay])) / len(success_rates_1_values_all[index_delay]))
    success_rates_total_values_average.append((1.0 * sum(success_rates_total_values_all[index_delay])) / len(success_rates_total_values_all[index_delay]))
print("\nsuccess_rates_0_values_average = ", [round(elem, 2) for elem in success_rates_0_values_average])
print("\nsuccess_rates_1_values_average = ", [round(elem, 2) for elem in success_rates_1_values_average])
print("\nsuccess_rates_total_values_average = ", [round(elem, 2) for elem in success_rates_total_values_average])

# figure_0 = plt.plot(max_allowed_delay_values, [100*elem for elem in success_rates_0_values_average])
# plt.xlabel("Maximum allowed delay (sec)")
# plt.ylabel("Success rate (%)")
# plt.title("Success rate of node 0 as a function of maximum allowed delay for a single channel")
# axes = plt.gca()
# axes.set_ylim([0, 100])
# plt.draw()
#
# figure_1 = plt.plot(max_allowed_delay_values, [100*elem for elem in success_rates_1_values_average])
# plt.xlabel("Maximum allowed delay (sec)")
# plt.ylabel("Success rate (%)")
# plt.title("Success rate of node 1 as a function of maximum allowed delay for a single channel")
# axes = plt.gca()
# axes.set_ylim([0, 100])
# plt.draw()
#
# figure_total = plt.plot(max_allowed_delay_values, [100*elem for elem in success_rates_total_values_average])
# plt.xlabel("Maximum allowed delay (sec)")
# plt.ylabel("Success rate (%)")
# plt.title("Overall success rate as a function of maximum allowed delay for a single channel")
# axes = plt.gca()
# axes.set_ylim([0, 100])
# plt.draw()



print(balance_history_node_0)

fig, ax = plt.subplots()
ax.plot(max_allowed_delay_values, [100*elem for elem in success_rates_0_values_average], label='Success rate of node 0')
ax.plot(max_allowed_delay_values, [100*elem for elem in success_rates_1_values_average], label='Success rate of node 1')
ax.plot(max_allowed_delay_values, [100*elem for elem in success_rates_total_values_average], label='Overall success rate')
ax.set_ylim([0, 100])
legend = ax.legend(loc='best')
plt.xlabel("Maximum allowed delay (sec)")
plt.ylabel("Success rate (%)")
plt.title("Success rate as a function of maximum allowed delay for a single channel")
plt.show()

# Plot balance history of node 0 from a single simulation
plt.figure(2)
plt.plot(balance_history_node_0[0][0][0], balance_history_node_0[0][0][1])
plt.xlabel("Simulation time")
plt.ylabel("Balance of node 0")
plt.title("Balance of node 0 for a single experiment ")
plt.show()
