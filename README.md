# Throughput-optimal scheduling in a payment channel

This package is a Python SimPy-based Discrete Event Simulator for payment scheduling in a payment channel.
It simulates a single payment channel with or without transaction buffers at each side, and allows for experiments with various transaction scheduling policies.
Transactions are generated from both sides according to customizable distributions of amounts and arrival times and are scheduled (processed) according to the chosen scheduling policy.

The user can choose:
* the initial channel balances
* the transaction generation parameters: total transactions from each side, amount distribution (constant, uniform, gaussian, pareto, empirical from csv file), interarrival time distribution (exponential with customizable parameter), deadline distribution (constant or uniform)
* the scheduling policy (`PMDE`, `PRI-IP`, `PRI-NIP`, or `PFI`, see paper for detailed explanation)
* the buffer discipline for the transaction buffers of the nodes: `oldest_first`, `youngest_first`, `closest_deadline_first`, `largest_amount_first`, `smallest_amount_first`
* buffering_capability for each node: `neither_node`, `only_node_0`, `only_node_1`, `both_separate`, `both_shared`
* the number of experiments over which to calculate the average metrics


The code accompanies the following paper: 

> N. Papadis and L. Tassiulas, "Payment Channel Networks: Single-Hop Scheduling for Throughput Maximization," IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, 2022, pp. 900-909, https://doi.org/10.1109/INFOCOM48880.2022.9796862.

The structure of the experiments performed in the paper and the relevant script for each experiment can be found in the file `experiments_structure.xlsx`.