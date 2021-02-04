import pypet
from simulate_channel import *
# import csv
# from statsmodels.distributions.empirical_distribution import ECDF

def pypet_wrapper(traj):
    node_0_parameters = [traj.initial_balance_0, traj.total_transactions_0, traj.exp_mean_0, traj.amount_distribution_0, traj.amount_distribution_parameters_0, traj.deadline_distribution_0]
    node_1_parameters = [traj.initial_balance_1, traj.total_transactions_1, traj.exp_mean_1, traj.amount_distribution_1, traj.amount_distribution_parameters_1, traj.deadline_distribution_1]

    results, all_transactions_list = simulate_channel(node_0_parameters, node_1_parameters,
                                                         traj.scheduling_policy, traj.buffer_discipline,
                                                         traj.who_has_buffer, traj.max_buffering_time,
                                                         traj.verbose, traj.seed)

    traj.f_add_result('success_count_node_0', results['success_counts'][0], comment='Number of successful transactions (node 0)')
    traj.f_add_result('success_count_node_1', results['success_counts'][1], comment='Number of successful transactions (node 1)')
    traj.f_add_result('success_count_channel_total', results['success_counts'][2], comment='Number of successful transactions (channel total)')
    traj.f_add_result('arrived_count_node_0', results['arrived_counts'][0], comment='Number of transactions that arrived (node 0)')
    traj.f_add_result('arrived_count_node_1', results['arrived_counts'][1], comment='Number of transactions that arrived (node 1)')
    traj.f_add_result('arrived_count_channel_total', results['arrived_counts'][2], comment='Number of transactions that arrived (channel total)')
    traj.f_add_result('throughput_node_0', results['throughputs'][0], comment='Throughput (Amount of successful transactions) (node 0)')
    traj.f_add_result('throughput_node_1', results['throughputs'][1], comment='Throughput (Amount of successful transactions) (node 1)')
    traj.f_add_result('throughput_channel_total', results['throughputs'][2], comment='Throughput (Amount of successful transactions) (channel total)')
    traj.f_add_result('arrived_amount_node_0', results['arrived_amounts'][0], comment='Amount of transactions that arrived (node 0)')
    traj.f_add_result('arrived_amount_node_1', results['arrived_amounts'][1], comment='Amount of transactions that arrived (node 1)')
    traj.f_add_result('arrived_amount_channel_total', results['arrived_amounts'][2], comment='Amount of transactions that arrived (channel total)')
    traj.f_add_result('sacrificed_count_node_0', results['sacrificed_counts'][0], comment='Number of sacrificed transactions (node 0)')
    traj.f_add_result('sacrificed_count_node_1', results['sacrificed_counts'][1], comment='Number of sacrificed transactions (node 1)')
    traj.f_add_result('sacrificed_count_channel_total', results['sacrificed_counts'][2], comment='Number of sacrificed transactions (channel total)')
    traj.f_add_result('sacrificed_amount_node_0', results['sacrificed_amounts'][0], comment='Amount of sacrificed transactions (node 0)')
    traj.f_add_result('sacrificed_amount_node_1', results['sacrificed_amounts'][1], comment='Amount of sacrificed transactions (node 1)')
    traj.f_add_result('sacrificed_amount_channel_total', results['sacrificed_amounts'][2], comment='Amount of sacrificed transactions (channel total)')
    traj.f_add_result('success_rate_node_0', results['success_rates'][0], comment='Success rate (node 0)')
    traj.f_add_result('success_rate_node_1', results['success_rates'][1], comment='Success rate (node 1)')
    traj.f_add_result('success_rate_channel_total', results['success_rates'][2], comment='Success rate (channel total)')
    traj.f_add_result('normalized_throughput_node_0', results['normalized_throughputs'][0], comment='Normalized throughput (node 0)')
    traj.f_add_result('normalized_throughput_node_1', results['normalized_throughputs'][1], comment='Normalized throughput (node 1)')
    traj.f_add_result('normalized_throughput_channel_total', results['normalized_throughputs'][2], comment='Normalized throughput (channel total)')

    traj.f_add_result('all_transactions_list', all_transactions_list, 'All transactions')



def main():
    # Create the environment
    env = pypet.Environment(trajectory='single_payment_channel_scheduling',
                            filename='./HDF5/results_87.hdf5',
                            overwrite_file=True)
    traj = env.traj
    # EMPIRICAL_DATA_FILEPATH = "./creditcard-non-fraudulent-only-amounts-only.csv"

    # SIMULATION PARAMETERS

    verbose = False
    num_of_experiments = 1

    # Node 0
    initial_balance_0 = 0
    total_transactions_0 = 500
    exp_mean_0 = 1 / 3
    # amount_distribution_0 = "constant"
    # amount_distribution_parameters_0 = [50]                                      # value of all transactions
    # amount_distribution_0 = "uniform"
    # amount_distribution_parameters_0 = [300]                               # max_transaction_amount
    amount_distribution_0 = "gaussian"
    amount_distribution_parameters_0 = [300, 100, 50]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]
    deadline_distribution_0 = "uniform"

    # Node 1
    initial_balance_1 = 300         # Capacity = 300
    total_transactions_1 = 500
    exp_mean_1 = 1 / 3
    # amount_distribution_1 = "constant"
    # amount_distribution_parameters_1 = [50]                                      # value of all transactions
    # amount_distribution_1 = "uniform"
    # amount_distribution_parameters_1 = [300]                               # max_transaction_amount
    amount_distribution_1 = "gaussian"
    amount_distribution_parameters_1 = [300, 100, 50]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]
    deadline_distribution_1 = "uniform"

    # if (amount_distribution_0 == "empirical_from_csv_file") or (amount_distribution_1 == "empirical_from_csv_file"):
    #     with open(EMPIRICAL_DATA_FILEPATH, newline='') as f:
    #         reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    #         empirical_data = list(reader)
    #         empirical_data = [x[0] for x in empirical_data]     # Convert to float from list
    #         ecdf = ECDF(empirical_data)
    #         if amount_distribution_0 == "empirical_from_csv_file":
    #             amount_distribution_parameters_0 = ecdf
    #         if amount_distribution_1 == "empirical_from_csv_file":
    #             amount_distribution_parameters_1 = ecdf
    #
    # print(ecdf)
    # exit(-1)
            
    # amount_distribution_0 = "constant"
    # amount_distribution_parameters_0 = [50]                                      # value of all transactions
    # amount_distribution_0 = "uniform"
    # amount_distribution_parameters_0 = [capacity]                               # max_transaction_amount
    # amount_distribution_0 = "gaussian"
    # amount_distribution_parameters_0 = [capacity, capacity/2, capacity/6]       # max_transaction_amount, gaussian_mean, gaussian_variance. E.g.: [capacity, capacity / 2, capacity / 6]
    # amount_distribution_0 = "pareto"
    # amount_distribution_parameters_0 = [1, 1.16, 1]                             # lower, shape, size
    # amount_distribution_0 = "powerlaw"
    # amount_distribution_parameters_0 = ...
    # amount_distribution_0 = "empirical_from_csv_file"

    # amount_distribution_1 = "constant"
    # amount_distribution_parameters_1 = [50]                                      # value of all transactions
    # amount_distribution_1 = "uniform"
    # amount_distribution_parameters_1 = [capacity]                               # max_transaction_amount
    # amount_distribution_1 = "gaussian"
    # amount_distribution_parameters_1 = [capacity, capacity/2, capacity/6]       # max_transaction_amount, gaussian_mean, gaussian_variance
    # amount_distribution_1 = "pareto"
    # amount_distribution_parameters_1 = [1, 1.16, 1]                             # lower, shape, size
    # amount_distribution_1 = "powerlaw"
    # amount_distribution_parameters_1 = ...
    # amount_distribution_1 = "empirical_from_csv_file"

    # deadline_distribution_0 = "constant"
    # deadline_distribution_0_parameters = [5]                # buffering time of all transactions
    # deadline_distribution_0 = "uniform"
    # # deadline_distribution_0_parameters = [5]                # max_buffering_time

    # deadline_distribution_1 = "constant"
    # deadline_distribution_1_parameters = [5]                # buffering time of all transactions
    # deadline_distribution_1 = "uniform"
    # # deadline_distribution_1_parameters = [5]                # max_buffering_time

    # Encode parameters for pypet

    traj.f_add_parameter('initial_balance_0', initial_balance_0, comment='Initial balance of node 0')
    traj.f_add_parameter('total_transactions_0', total_transactions_0, comment='Total transactions arriving at node 0')
    traj.f_add_parameter('exp_mean_0', exp_mean_0, comment='Rate of exponentially distributed arrivals at node 0')
    # traj.f_add_parameter('max_transaction_amount_0', traj.initial_balance_0 + traj.initial_balance_1,
    #                      comment='Maximum possible amount for incoming transactions at node 0')
    traj.f_add_parameter('amount_distribution_0', amount_distribution_0, comment='The distribution of the transaction amounts at node 0')
    traj.f_add_parameter('amount_distribution_parameters_0', amount_distribution_parameters_0, comment='Parameters of the distribution of the transaction amounts at node 0')
    traj.f_add_parameter('deadline_distribution_0', deadline_distribution_0, comment='The distribution of the transaction deadlines at node 0')
    # traj.f_add_parameter('deadline_distribution_0_parameters', deadline_distribution_0_parameters, comment='Parameters of the distribution of the transaction deadlines at node 0')

    traj.f_add_parameter('initial_balance_1', initial_balance_1, comment='Initial balance of node 1')
    traj.f_add_parameter('total_transactions_1', total_transactions_1, comment='Total transactions arriving at node 1')
    traj.f_add_parameter('exp_mean_1', exp_mean_1, comment='Rate of exponentially distributed arrivals at node 1')
    # traj.f_add_parameter('max_transaction_amount_1', traj.initial_balance_0 + traj.initial_balance_1,
    #                      comment='Maximum possible amount for incoming transactions at node 1')
    traj.f_add_parameter('amount_distribution_1', amount_distribution_1, comment='The distribution of the transaction amounts at node 1')
    traj.f_add_parameter('amount_distribution_parameters_1', amount_distribution_parameters_1, comment='Parameters of the distribution of the transaction amounts at node 1')
    traj.f_add_parameter('deadline_distribution_1', deadline_distribution_1, comment='The distribution of the transaction deadlines at node 1')
    # traj.f_add_parameter('deadline_distribution_1_parameters', deadline_distribution_1_parameters, comment='Parameters of the distribution of the transaction deadlines at node 1')

    traj.f_add_parameter('scheduling_policy', "PMDE", comment='Scheduling policy')
    traj.f_add_parameter('buffer_discipline', "oldest_first", comment='Order of processing transactions in the buffer')
    traj.f_add_parameter('who_has_buffer', "none", comment='Which node has a buffer')
    traj.f_add_parameter('max_buffering_time', 0, comment='Maximum time before a transaction expires')

    traj.f_add_parameter('verbose', verbose, comment='Verbose output')
    traj.f_add_parameter('num_of_experiments', num_of_experiments, comment='Repetitions of every experiment')
    traj.f_add_parameter('seed', 0, comment='Randomness seed')

    seeds = [63621, 87563, 24240, 14020, 84331, 60917, 48692, 73114, 90695, 62302, 52578, 43760, 84941, 30804, 40434, 63664, 25704, 38368, 45271, 34425]

    traj.f_explore(pypet.cartesian_product({
                                            'scheduling_policy': ["PMDE", "PRI-IP", "PRI-NIP"],
                                            'buffer_discipline': ["oldest_first", "youngest_first", "closest_deadline_first", "largest_amount_first", "smallest_amount_first"],
                                            # 'buffer_discipline': ["largest_amount_first"],
                                            # 'who_has_buffer': ["none", "only_node_0", "only_node_1", "both_separate", "both_shared"],
                                            'who_has_buffer': ["both_shared"],
                                            # 'max_buffering_time': [5],
                                            # 'max_buffering_time': range(0,300,50),
                                            'max_buffering_time': list(range(1, 10, 1)) + list(range(10, 120, 10)),
                                            'seed': seeds[1:traj.num_of_experiments + 1]}))

    # Run wrapping function instead of simulator directly
    env.run(pypet_wrapper)


if __name__ == '__main__':
    main()
