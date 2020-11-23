import numpy as np
import pypet
from DES.sc_DES_with_all_kinds_of_buffers_fun_pypet import sc_DES_with_all_kinds_of_buffers_fun
import pandas as pd


def pypet_wrapper(traj):
    sr, sa, bh0, sac, atl = sc_DES_with_all_kinds_of_buffers_fun([traj.initial_balance_0, traj.initial_balance_1],
                                                                 traj.total_transactions_0,
                                                                 traj.max_transaction_amount_0, traj.exp_mean_0,
                                                                 traj.total_transactions_1,
                                                                 traj.max_transaction_amount_1, traj.exp_mean_1,
                                                                 traj.max_buffering_time,
                                                                 traj.who_has_buffer, traj.immediate_processing,
                                                                 traj.processing_order,
                                                                 traj.verbose, traj.seed)
    traj.f_add_result('success_rate_0', sr[0], comment='Success rate of node 0')
    traj.f_add_result('success_rate_1', sr[1], comment='Success rate of node 1')
    traj.f_add_result('success_rate_total', sr[2], comment='Total channel success rate')
    traj.f_add_result('throughput_0', sa[0], comment='Throughput for node 0')
    traj.f_add_result('throughput_1', sa[1], comment='Throughput for node 1')
    traj.f_add_result('balance_history_node_0_times', bh0[0], comment='Times when balance history of node 0 changes')
    traj.f_add_result('balance_history_node_0_values', bh0[1], comment='Balance history of node 0')
    traj.f_add_result('sacrificed_0', int(sac[0]), comment='Number of sacrificed transactions for node 0')
    traj.f_add_result('sacrificed_1', int(sac[1]), comment='Number of sacrificed transactions for node 1')
    traj.f_add_result('all_transactions_list', atl, 'All transactions')
    # traj.f_add_result('all_transactions_list_node_0', atl[0], 'All transactions for node 0')
    # traj.f_add_result('all_transactions_list_node_1', atl[1], 'All transactions for node 1')
    # traj.f_add_result('all_transactions_list_node_0', pd.DataFrame(atl[0]), 'All transactions for node 0')
    # traj.f_add_result('all_transactions_list_node_1', pd.DataFrame(atl[1]), 'All transactions for node 1')


def main():
    np.set_printoptions(precision=2)

    # Create the environment
    env = pypet.Environment(trajectory='single_channel_buffering',
                            filename='./HDF5/results.hdf5',
                            overwrite_file=True)
    traj = env.traj

    # SIMULATION PARAMETERS
    traj.f_add_parameter('initial_balance_0', 150, comment='Initial balance of node 0')
    traj.f_add_parameter('initial_balance_1', 150, comment='Initial balance of node 1')

    traj.f_add_parameter('total_transactions_0', 200, comment='Total transactions arriving at node 0')
    traj.f_add_parameter('max_transaction_amount_0', traj.initial_balance_0 + traj.initial_balance_1,
                         comment='Maximum possible amount for incoming transactions at node 0')
    traj.f_add_parameter('exp_mean_0', 1 / 3, comment='Rate of exponentially distributed arrivals at node 0')
    traj.f_add_parameter('total_transactions_1', 200, comment='Total transactions arriving at node 1')
    traj.f_add_parameter('max_transaction_amount_1', traj.initial_balance_0 + traj.initial_balance_1,
                         comment='Maximum possible amount for incoming transactions at node 1')
    traj.f_add_parameter('exp_mean_1', 1 / 3, comment='Rate of exponentially distributed arrivals at node 1')

    traj.f_add_parameter('who_has_buffer', "none", comment='Which node has a buffer')
    traj.f_add_parameter('immediate_processing', True,
                         comment='Immediate processing of incoming transactions if feasible')
    traj.f_add_parameter('processing_order', "oldest_transaction_first",
                         comment='Order of processing transactions in the buffer')
    traj.f_add_parameter('max_buffering_time', 0, comment='Maximum time before a transaction expires')

    traj.f_add_parameter('verbose', True, comment='Verbose output')
    traj.f_add_parameter('num_of_experiments', 1, comment='Repetitions of every experiment')
    traj.f_add_parameter('seed', 0, comment='Randomness seed')

    seeds = [63621, 87563, 24240, 14020, 84331, 60917, 48692, 73114, 90695, 62302, 52578, 43760, 84941, 30804, 40434,
             63664, 25704, 38368, 45271, 34425]

    traj.f_explore(pypet.cartesian_product({
                                            # 'who_has_buffer': ["none", "only_node_0", "only_node_1", "both_separate", "both_shared"],
                                            'who_has_buffer': ["both_shared"],
                                            'immediate_processing': [True],
                                            'processing_order': ["oldest_transaction_first", "closest_deadline_first"],
                                            # 'processing_order': ["oldest_transaction_first", "youngest_transaction_first", "closest_deadline_first", "largest_amount_first", "smallest_amount_first"],
                                            'max_buffering_time': range(0,100,50),
                                            # 'max_buffering_time': list(range(0, 50, 5)) + list(range(50, 600+1, 50)),
                                            'seed': seeds[1:traj.num_of_experiments + 1]}))

    # Run wrapping function instead of simulator directly
    env.run(pypet_wrapper)


if __name__ == '__main__':
    main()
