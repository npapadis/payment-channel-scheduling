# Throughput-optimal scheduling in a payment channel

This package is a Python SimPy-based Discrete Event Simulator for payment scheduling in a payment channel.
It simulates a single payment channel with or without transaction buffers at each side, and allows for experiments with various transaction scheduling policies.
Transactions are generated from both sides according to customizable distributions of amounts and arrival times and are scheduled (processed) according to the chosen scheduling policy.

The user can choose:
* the initial channel balances
* the transaction generation parameters: total transactions from each side, amount distribution (constant, uniform, gaussian, pareto, empirical from csv file), arrival time distribution (exponential with customizable parameter), deadline distribution (constant or uniform)
* the scheduling policy (PMDE, PRI-IP, PRI-NIP, or PFI, see paper for detailed explanation)
* the buffer discipline for the transaction buffers of the nodes: oldest_first, youngest_first, closest_deadline_first, largest_amount_first, smallest_amount_first
* buffering_capability for each node: neither_node, only_node_0, only_node_1, both_separate, both_shared
* the number of experiments of which to calculate the average metrics

This code accompanies the paper "Payment Channel Networks: Single-Hop Scheduling for Throughput Maximization" by N. Papadis and L. Tassiulas which will appear in the proceedings of the [IEEE International Conference on Computer Communications (INFOCOM) 2022](https://infocom2022.ieee-infocom.org).
