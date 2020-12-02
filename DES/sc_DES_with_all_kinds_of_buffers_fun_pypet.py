"""
This script uses Discrete Event Simulation via SimPy to simulate a single payment channel.
It is possible that a buffer exists and accepts transactions that are not executed immediately. There are 5 options:
    - None of the nodes has a buffer
    - Only node 0 has a buffer
    - Only node 1 has a buffer
    - Each of nodes 0 and 1 has its own buffer
    - Both nodes have a shared buffer
Transactions are generated from both sides according to customizable distributions of amounts and arrival times.
A transaction arriving at the channel is processed immediately, if possible.
Otherwise, if a buffer exists in the corresponding origin node, it is added to the buffer until it processed successfully or its deadline expires.
The buffer processes the transactions according to the algorithm:
    1) oldest transaction first
    2) shortest deadline first
    3) largest amount first

Max buffering time is defined as a transaction property, not a channel property.
The buffer is processed every 1 second until the total simulation time has been reached.

To-do (?): Terminate simulation at maximum simulation time, not when a certain number of transactions is generated,
to avoid rejecting the remaining transactions in the buffers, which could have been successful if more transactions were arriving.
"""

from numpy import random
import simpy
import sys
# import powerlaw
import pandas as pd
import sortedcontainers as sc

# sc.SortedKeyList.__repr__ = lambda skl: list(skl)


class Transaction:
    def __init__(self, env, channel, time, from_node, to_node, amount, max_buffering_time, verbose):
        self.env = env
        self.channel = channel
        self.time = time
        self.from_node = from_node
        self.to_node = to_node
        self.amount = amount
        self.max_buffering_time = max_buffering_time
        self.verbose = verbose
        self.buffered = False
        self.status = "PENDING"     # Other statuses: "SUCCEEDED", "REJECTED", "EXPIRED"
        self.initially_feasible = None
        self.request = None
        self.preemptied = self.env.event()

        if self.verbose:
            print("Transaction {} generated at time {:.2f}.".format(self, self.env.now))

        # Start the run process every time an instance is created.
        # env.process(self.run())

    def run(self):
        if self.status == "PENDING":
            with self.channel.channel_link.request() as request:    # Generate a request event
                yield request                                       # Wait for access to the channel
                self.request = request
                yield self.env.process(self.channel.process_transaction(self))              # Once the channel belongs to the transaction, try to process it.

    def __repr__(self):
        return "%d->%d t=%.2f D=%d a=%d" % (self.from_node, self.to_node, self.time, self.max_buffering_time, self.amount)


# class BufferedTransaction:
#     def __init__(self, t, processing_order):
#         self.t = t
#         self.processing_order = processing_order
#
#     def _cmp_key(self):
#         return self.t, self.processing_order
#
#     def __eq__(self, other):
#         # noinspection PyProtectedMember
#         return self._cmp_key() == other._cmp_key()
#
#     # def __lt__(self, other):
#     #     return self._cmp_key() < other._cmp_key()
#
#     def __lt__(self, other):
#         if self.processing_order == "oldest_transaction_first":
#             return self.t.time >= other.t.time
#         elif self.processing_order == "youngest_transaction_first":
#             return self.t.time <= other.t.time
#         elif self.processing_order == "closest_deadline_first":
#             return self.t.time + self.t.max_buffering_time <= other.t.time + other.t.max_buffering_time
#         elif self.processing_order == "largest_amount_first":
#             return self.t.amount >= other.t.amount
#         elif self.processing_order == "smallest_amount_first":
#             return self.t.amount <= other.t.amount
#         # elif optimal....
#         else:
#             print("Input error: {} is not a valid 'processing_order' value.".format(self.processing_order))
#             sys.exit(1)
#
#     def __repr__(self):
#         print(self.t)


class Channel:

    def __init__(self, env, node0, node1, capacity, balances, who_has_buffer, immediate_processing, processing_order, verbose,
                 total_simulation_time_estimation):
        self.env = env
        self.node0 = node0
        self.node1 = node1
        self.capacity = capacity
        self.balances = balances
        self.immediate_processing = immediate_processing
        self.verbose = verbose
        self.channel_link = simpy.Resource(env, capacity=1)
        self.successful_transactions = [0, 0]
        self.successful_amounts = [0, 0]
        self.balance_history_node_0_times = []
        self.balance_history_node_0_values = []

        if who_has_buffer == "none":
            self.buffers = [None, None]
        elif who_has_buffer == "only_node_0":
            self.buffers = [Buffer(env, node0, self, processing_order, verbose, total_simulation_time_estimation), None]
            self.env.process(self.buffers[0].run())
        elif who_has_buffer == "only_node_1":
            self.buffers = [None, Buffer(env, node1, self, processing_order, verbose, total_simulation_time_estimation)]
            self.env.process(self.buffers[1].run())
        elif who_has_buffer == "both_separate":
            self.buffers = [Buffer(env, node0, self, processing_order, verbose, total_simulation_time_estimation),
                            Buffer(env, node1, self, processing_order, verbose, total_simulation_time_estimation)]
            self.env.process(self.buffers[0].run())
            self.env.process(self.buffers[1].run())
        elif who_has_buffer == "both_shared":
            shared_buffer = Buffer(env, node0, self, processing_order, verbose, total_simulation_time_estimation)
            self.buffers = [shared_buffer, shared_buffer]
            self.env.process(self.buffers[0].run())
        else:
            print("Input error: {} is not a valid 'who_has_buffer' value.".format(who_has_buffer))
            sys.exit(1)

    # def process_transaction(self, t):
    #     if ((t.buffered is False) and (self.immediate_processing is True)) or (t.buffered is True) or (self.buffers[t.from_node] is None):
    #                                                                                                     # if (first time and immediate processing enabled) or (not first time), then
    #         if t.amount <= self.balances[t.from_node]:                                                      # if feasible, then process
    #             self.balances[t.from_node] -= t.amount
    #             self.balances[t.to_node] += t.amount
    #             self.successful_transactions[t.from_node] += 1
    #             self.successful_amounts[t.from_node] += t.amount
    #             self.balance_history_node_0_times.append(self.env.now)
    #             self.balance_history_node_0_values.append(self.balances[0])
    #             if self.verbose:
    #                 print("Transaction {} processed at time {:.2f}.".format(t, self.env.now))
    #                 print("New balances are", self.balances)
    #             return True
    #         elif (self.buffers[t.from_node] is not None) and (t.buffered is False):                         # if not feasible and first time and buffer exists, then add to buffer
    #             self.buffers[t.from_node].transaction_list.append(t)
    #             # print(self.buffers[t.from_node].transaction_list)
    #             t.buffered = True
    #             if self.verbose:
    #                 print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
    #                 print("Unchanged balances are", self.balances)
    #                 if self.buffers[0] is not None: print("Buffer 0:", self.buffers[0].transaction_list)
    #                 if self.buffers[1] is not None: print("Buffer 1:", self.buffers[1].transaction_list)
    #         elif self.buffers[t.from_node] is None:                                                         # if not feasible and no buffer exists, then reject
    #             if self.verbose:
    #                 print("Transaction {} rejected at time {:.2f}.".format(t, self.env.now))
    #                 print("Unchanged balances are", self.balances)
    #         else:                                                                                           # if not feasible and not first time, then leave in buffer
    #             pass
    #     else:  # Add to buffer                                                                          # if first time and immediate processing disabled, then
    #         if (t.buffered is False) and (self.buffers[t.from_node] is not None):                           # if buffer exists, then add to it
    #             self.buffers[t.from_node].transaction_list.append(t)
    #             # print(self.buffers[t.from_node].transaction_list)
    #             t.buffered = True
    #             if self.verbose:
    #                 print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
    #                 print("Unchanged balances are", self.balances)
    #                 if self.buffers[0] is not None: print("Buffer 0:", self.buffers[0].transaction_list)
    #                 if self.buffers[1] is not None: print("Buffer 1:", self.buffers[1].transaction_list)
    #         else:                                                                                           # if buffer does not exist, then reject
    #             if self.verbose:
    #                 print("Transaction {} rejected at time {:.2f}.".format(t, self.env.now))
    #                 print("Unchanged balances are", self.balances)

    # WITH IMMEDIATE PROCESSING BY DEFAULT
    # def process_transaction(self, t):
    #     if t.amount <= self.balances[t.from_node]:
    #         self.balances[t.from_node] -= t.amount
    #         self.balances[t.to_node] += t.amount
    #         self.successful_transactions[t.from_node] += 1
    #         self.successful_amounts[t.from_node] += t.amount
    #         self.balance_history_node_0_times.append(self.env.now)
    #         self.balance_history_node_0_values.append(self.balances[0])
    #         if self.verbose:
    #             print("Transaction {} processed at time {:.2f}.".format(t, self.env.now))
    #             print("New balances are", self.balances)
    #         return True
    #     elif (self.buffers[t.from_node] is None) and self.verbose:
    #             print("Transaction {} rejected at time {:.2f}.".format(t, self.env.now))
    #             print("Unchanged balances are", self.balances)
    #     else:  # Add to buffer
    #         if (t.buffered is False) and (self.buffers[t.from_node] is not None):
    #             self.buffers[t.from_node].transaction_list.append(t)
    #             # print(self.buffers[t.from_node].transaction_list)
    #             t.buffered = True
    #             if self.verbose:
    #                 print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
    #                 print("Unchanged balances are", self.balances)
    #                 if self.buffers[0] is not None: print(self.buffers[0].transaction_list)
    #                 if self.buffers[1] is not None: print(self.buffers[1].transaction_list)

    def execute_transaction(self, t):
        # Calling this function requires checking for transaction feasibility beforehand. The function itself does not perform any checks, and this could lead to negative balances if misused.

        FT = t.buffered is False  # First Time

        self.balances[t.from_node] -= t.amount
        self.balances[t.to_node] += t.amount
        self.successful_transactions[t.from_node] += 1
        self.successful_amounts[t.from_node] += t.amount
        self.balance_history_node_0_times.append(self.env.now)
        self.balance_history_node_0_values.append(self.balances[0])

        if self.verbose:
            if FT:
                print("Transaction {} processed at time {:.2f}.".format(t, self.env.now))
            else:
                print("SUCCESS: Transaction {} was processed and removed from buffer at time {:.2f}.".format(t, self.env.now))
            print("New balances are", self.balances)

        t.status = "SUCCEEDED"

    def reject_transaction(self, t):
        FT = t.buffered is False  # First Time

        if self.verbose:
            if FT:
                print("Transaction {} rejected at time {:.2f}.".format(t, self.env.now))
                print("Unchanged balances are", self.balances)
            else:
                print("FAILURE: Transaction {} expired and was removed from buffer at time {:.2f}.".format(t, self.env.now))
        t.status = "REJECTED"

    def add_transaction_to_buffer(self, t):
        # self.buffers[t.from_node].transaction_list.append(t)

        # processing_order = self.buffers[t.from_node].processing_order
        # self.buffers[t.from_node].transaction_list.append(BufferedTransaction(t, processing_order))
        self.buffers[t.from_node].transaction_list.add(t)

        # print(self.buffers[t.from_node].transaction_list)
        t.buffered = True
        if self.verbose:
            print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
            print("Unchanged balances are", self.balances)
            if self.buffers[0] is not None: print("Buffer 0:", list(self.buffers[0].transaction_list))
            if self.buffers[1] is not None: print("Buffer 1:", list(self.buffers[1].transaction_list))
        # t.status = "PENDING"  # t.status is "PENDING" already

    def process_transaction(self, t):

        IP = self.immediate_processing is True          # Immediate Processing
        BE = self.buffers[t.from_node] is not None      # Buffer Exists
        FT = t.buffered is False                        # First Time
        FE = t.amount <= self.balances[t.from_node]     # FEasible
        # Configurations "not BE and not FT" are not reachable. The remaining 12 of the 16 configurations are covered below.

        if t.status != "PENDING":
            print("Error in process_transaction(): attempt to process non-pending transaction (of status \"{}\").".format(t.status))
            sys.exit(1)

        if BE and self.buffers[t.from_node].processing_order == "optimal_policy":      # optimal policy
            if not BE and FE:   # process
                self.execute_transaction(t)
            elif BE:
                self.add_transaction_to_buffer(t)
                deadline = t.time + t.max_buffering_time - self.env.now
                self.channel_link.release(t.request)
                resume_reason = yield self.env.timeout(deadline) | t.preemptied
                if t.preemptied in resume_reason:
                    return True
                else:
                    with self.channel_link.request() as request:
                        resume_reason = yield request | t.preemptied
                        if request not in resume_reason:
                            return True
                        else:
                            t.request = request
                            FE_upon_expiration = t.amount <= self.balances[t.from_node]
                            if FE_upon_expiration:
                                self.buffers[t.from_node].transaction_list.remove(t)
                                self.execute_transaction(t)
                                if self.verbose:
                                    if self.buffers[0] is not None: print("Buffer 0:", list(self.buffers[0].transaction_list))
                                    if self.buffers[1] is not None: print("Buffer 1:", list(self.buffers[1].transaction_list))
                                return True
                            else:
                                if t.amount <= self.balances[t.to_node] and self.buffers[t.to_node].transaction_list:
                                    # Version 1: policy for all transaction amounts equal
                                    opposite_tx = self.buffers[t.to_node].transaction_list.pop(index=0)
                                    opposite_tx.preemptied.succeed()
                                    if self.verbose:
                                        print("PREEMPTION FOLLOWING:")
                                    self.execute_transaction(opposite_tx)
                                    self.buffers[t.from_node].transaction_list.remove(t)
                                    self.execute_transaction(t)
                                    if self.verbose:
                                        if self.buffers[0] is not None: print("Buffer 0:", list(self.buffers[0].transaction_list))
                                        if self.buffers[1] is not None: print("Buffer 1:", list(self.buffers[1].transaction_list))
                                    return True
                                    # # Version 2: policy for unequal amounts (WARNING: this might result in invalid state without proper checks)
                                    # total_opposite_amount = 0
                                    # while total_opposite_amount < t.amount and self.buffers[t.to_node].transaction_list:
                                    #     opposite_tx = self.buffers[t.to_node].transaction_list.pop(index=0)
                                else:
                                    self.buffers[t.from_node].transaction_list.remove(t)
                                    self.reject_transaction(t)
                                    if self.verbose:
                                        if self.buffers[0] is not None: print("Buffer 0:", list(self.buffers[0].transaction_list))
                                        if self.buffers[1] is not None: print("Buffer 1:", list(self.buffers[1].transaction_list))
                                    return False

            else:   # reject
                self.reject_transaction(t)
                return False

        else:                                                                   # heuristic policy
            if FT and FE:
                t.initially_feasible = True
            else:
                t.initially_feasible = False

            if (IP and BE and FT and FE) or (IP and BE and not FT and FE) or (IP and not BE and FT and FE) or (
                    not IP and BE and not FT and FE):  # process
                    # Once the channel belongs to the transaction, then if the deadline has not expired, try to process it.
                    if t.time + t.max_buffering_time >= t.env.now:
                        self.execute_transaction(t)
                        return True
                    else:   # Transaction expired and will be handled in the next processing of the buffer.
                        return False
            elif (IP and BE and FT and not FE) or (not IP and BE and FT):  # add to buffer
                self.add_transaction_to_buffer(t)
                return False
            elif (IP and not BE and FT and not FE) or (not IP and not BE and FT):  # reject
                self.reject_transaction(t)
                return False
            elif BE and not FT and not FE:  # skip
                pass
                # t.status = "PENDING"  # t.status is "PENDING" already
                return False
            else:
                print("Unreachable state reached. Exiting.")
                # self.channel_link.release(request)
                sys.exit(1)



class Buffer:
    def __init__(self, env, node, channel, processing_order, verbose, total_simulation_time_estimation):
        self.env = env
        self.node = node
        self.channel = channel
        # self.max_buffering_time = max_buffering_time
        self.processing_order = processing_order
        self.verbose = verbose
        self.total_simulation_time_estimation = total_simulation_time_estimation
        # self.total_successes = 0

        if self.processing_order == "oldest_transaction_first":
            key = lambda t: t.time
        elif self.processing_order == "youngest_transaction_first":
            key = lambda t: - t.time
        elif self.processing_order in ["closest_deadline_first", "optimal_policy"]:
            key = lambda t: t.time + t.max_buffering_time
        elif self.processing_order == "largest_amount_first":
            key = lambda t: t.amount
        elif self.processing_order == "smallest_amount_first":
            key = lambda t: - t.amount
        # elif optimal....
        else:
            print("Input error: {} is not a valid 'processing_order' value.".format(self.processing_order))
            sys.exit(1)

        self.transaction_list = sc.SortedKeyList(key=key)
        # self.transaction_list.__repr__ = lambda skl: list(skl)

    def run(self):
        if self.processing_order in ["oldest_transaction_first", "youngest_transaction_first", "closest_deadline_first", "largest_amount_first", "smallest_amount_first"]:
            while self.env.now <= self.total_simulation_time_estimation:
                # s = self.process_buffer()
                # s = self.process_buffer_greedy()
                yield self.env.process(self.process_buffer_greedy())
                # self.total_successes = self.total_successes + s
                yield self.env.timeout(1)

    # def process_buffer(self):    # returns total successful transactions
    #     total_successes_this_time = 0
    #     some_transaction_was_processed_with_current_buffer = 1
    #
    #     while some_transaction_was_processed_with_current_buffer == 1:
    #         some_transaction_was_processed_with_current_buffer = 0
    #         for t in self.transaction_list:
    #             if t.time + t.max_buffering_time < self.env.now:  # if t is too old, drop it completely
    #                 self.transaction_list.remove(t)
    #                 if self.verbose:
    #                     print("FAILURE: Transaction {} expired and was removed from buffer at time {:.2f}.".format(t, self.env.now))
    #                     if self.channel.buffers[0] is not None: print("Buffer 0:", self.channel.buffers[0].transaction_list)
    #                     if self.channel.buffers[1] is not None: print("Buffer 1:", self.channel.buffers[1].transaction_list)
    #
    #             else:  # if t is not too old and can be processed, process it
    #                 if self.channel.process_transaction(t):
    #                     self.transaction_list.remove(t)
    #                     if self.verbose:
    #                         print("SUCCESS: Transaction {} was processed and removed from buffer at time {:.2f}.".format(t, self.env.now))
    #                         print("New balances are", self.channel.balances)
    #                         if self.channel.buffers[0] is not None: print("Buffer 0:", self.channel.buffers[0].transaction_list)
    #                         if self.channel.buffers[1] is not None: print("Buffer 1:", self.channel.buffers[1].transaction_list)
    #                     some_transaction_was_processed_with_current_buffer = 1
    #                     total_successes_this_time += 1
    #                 else:
    #                     pass
    #         # if no_more_incoming_transactions:
    #         #     current_time += max_buffering_time / 10
    #         # if self.verbose:
    #         #     print("current time: ", current_time)
    #
    #     return total_successes_this_time

    def process_buffer_greedy(self):
        # Processes all transactions that are possible now and returns total successful transactions.
        total_successes_this_time = 0

        for t in self.transaction_list:  # while list not empty
            # t = self.transaction_list.pop(index=0)
            if t.time + t.max_buffering_time < self.env.now:  # if t is too old, reject it and remove it from buffer
                t.status = "EXPIRED"
                self.transaction_list.remove(t)
                if self.verbose:
                    print("FAILURE: Transaction {} expired and was removed from buffer at time {:.2f}.".format(t, self.env.now))
                    if self.channel.buffers[0] is not None: print("Buffer 0:", list(self.channel.buffers[0].transaction_list))
                    if self.channel.buffers[1] is not None: print("Buffer 1:", list(self.channel.buffers[1].transaction_list))
            else:  # if t is not too old and can be processed, process it
                # if self.channel.process_transaction(t):
                yield self.env.process(t.run())
                if t.status == "SUCCEEDED":
                    self.transaction_list.remove(t)
                    if self.verbose:
                        # print("SUCCESS: Transaction {} was processed and removed from buffer at time {:.2f}.".format(t, self.env.now))
                        # print("New balances are", self.channel.balances)
                        if self.channel.buffers[0] is not None: print("Buffer 0:",
                                                                      list(self.channel.buffers[0].transaction_list))
                        if self.channel.buffers[1] is not None: print("Buffer 1:",
                                                                      list(self.channel.buffers[1].transaction_list))
                    total_successes_this_time += 1
                else:
                    pass

        return total_successes_this_time

    # def process_buffer_optimal(self):


    # def __repr__(self):
    #     return list(self.transaction_list)


def transaction_generator(env, channel, from_node, total_transactions, exp_mean, amount_distribution, amount_distribution_parameters,
                                      deadline_distribution, max_buffering_time, all_transactions_list, verbose):
    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        to_node = 1 if (from_node == 0) else 0

        if amount_distribution == "constant":
            amount = amount_distribution_parameters[0]
        elif amount_distribution == "uniform":
            max_transaction_amount = amount_distribution_parameters[0]
            amount = random.randint(1, max_transaction_amount)
        elif amount_distribution == "gaussian":
            max_transaction_amount = amount_distribution_parameters[0]
            gaussian_mean = amount_distribution_parameters[1]
            gaussian_variance = amount_distribution_parameters[2]
            amount = round(max(1, min(max_transaction_amount, random.normal(gaussian_mean, gaussian_variance))))
        elif amount_distribution == "pareto":
            lower = amount_distribution_parameters[0]  # the lower end of the support
            shape = amount_distribution_parameters[1]  # the distribution shape parameter, also known as `a` or `alpha`
            size = amount_distribution_parameters[2]  # the size of your sample (number of random values)
            amount = random.pareto(shape, size) + lower
        # elif amount_distribution == "powerlaw":
            # powerlaw.Power_Law(xmin=1, xmax=2, discrete=True, parameters=[1.16]).generate_random(n=10)
        else:
            print("Input error: {} is not a supported amount distribution or the parameters {} given are invalid.".format(amount_distribution, amount_distribution_parameters))
            sys.exit(1)

        # Distribution for max_buffering_time
        if deadline_distribution == "constant":
            # max_buffering_time = deadline_distribution_parameters[0]
            t = Transaction(env, channel, env.now, from_node, to_node, amount, max_buffering_time, verbose)
        elif deadline_distribution == "uniform":
            # max_buffering_time = deadline_distribution_parameters[0]
            initial_deadline = random.randint(0, max_buffering_time) if max_buffering_time > 0 else 0
            t = Transaction(env, channel, env.now, from_node, to_node, amount, initial_deadline, verbose)
        else:
            # print("Input error: {} is not a supported deadline distribution or the parameters {} given are invalid.".format(deadline_distribution, deadline_distribution_parameters))
            print("Input error: {} is not a supported deadline distribution.".format(deadline_distribution))
            sys.exit(1)

        all_transactions_list.append(t)

        env.process(t.run())
        # env.process(transaction(env, env.now, from_node, to_node, amount))
        # env.process(transaction(env, channel, env.now, from_node, to_node, amount))
        # print(Transaction(env, env.now, from_node, to_node, amount))
        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)



def sc_DES_with_all_kinds_of_buffers_fun(initial_balances,
                                         total_transactions_0, exp_mean_0, amount_distribution_0, amount_distribution_0_parameters, deadline_distribution_0, max_buffering_time_0,
                                         total_transactions_1, exp_mean_1, amount_distribution_1, amount_distribution_1_parameters, deadline_distribution_1, max_buffering_time_1,
                                         who_has_buffer, immediate_processing, processing_order,
                                         verbose, seed):
    total_simulation_time_estimation = 2 * max(total_transactions_0 * 1 / exp_mean_0,
                                               total_transactions_1 * 1 / exp_mean_1)
    random.seed(seed)

    env = simpy.Environment()

    channel = Channel(env, 0, 1, sum(initial_balances), initial_balances, who_has_buffer, immediate_processing, processing_order, verbose,
                      total_simulation_time_estimation)

    all_transactions_list = []
    env.process(transaction_generator(env, channel, 0, total_transactions_0, exp_mean_0, amount_distribution_0, amount_distribution_0_parameters,
                                      deadline_distribution_0, max_buffering_time_0, all_transactions_list, verbose))
    env.process(transaction_generator(env, channel, 1, total_transactions_1, exp_mean_1, amount_distribution_1, amount_distribution_1_parameters,
                                      deadline_distribution_1, max_buffering_time_1, all_transactions_list, verbose))
    # env.process(transaction_generator(env, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0))
    # env.process(transaction_generator(env, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1))

    env.run()

    success_rates = [channel.successful_transactions[0] / total_transactions_0,
                     channel.successful_transactions[1] / total_transactions_1,
                     (channel.successful_transactions[0] + channel.successful_transactions[1]) / (
                                 total_transactions_0 + total_transactions_1)]

    # sacrificed_0 = sum(1 for t in all_transactions_list_node_0 if (t.initially_feasible is True and t.status in ["REJECTED", "EXPIRED"]))
    # sacrificed_1 = sum(1 for t in all_transactions_list_node_1 if (t.initially_feasible is True and t.status in ["REJECTED", "EXPIRED"]))
    sacrificed_0 = sum(1 for t in all_transactions_list if (t.initially_feasible is True and t.status in ["REJECTED", "EXPIRED"]))
    sacrificed_1 = sum(1 for t in all_transactions_list if (t.initially_feasible is True and t.status in ["REJECTED", "EXPIRED"]))

    if verbose:
        print("Success rate:", success_rates)
        print("Total successfully processed amounts:", channel.successful_amounts)

        if channel.buffers[0] is not None: print("Buffer 0:", list(channel.buffers[0].transaction_list))
        if channel.buffers[1] is not None: print("Buffer 1:", list(channel.buffers[1].transaction_list))

    for t in all_transactions_list:
        del t.env
        del t.channel
        del t.request
    # all_transactions_list_node_0 = [vars(t) for t in all_transactions_list_node_0]
    all_transactions_list = pd.DataFrame([vars(t) for t in all_transactions_list])
    # for t in all_transactions_list_node_0:
    #     del t.env
    #     del t.channel
    # # all_transactions_list_node_0 = [vars(t) for t in all_transactions_list_node_0]
    # all_transactions_list_node_0 = pd.DataFrame([vars(t) for t in all_transactions_list_node_0])
    # for t in all_transactions_list_node_1:
    #     del t.env
    #     del t.channel
    # # all_transactions_list_node_1 = [vars(t) for t in all_transactions_list_node_1]
    # all_transactions_list_node_1 = pd.DataFrame([vars(t) for t in all_transactions_list_node_1])

    return success_rates, \
           channel.successful_amounts, \
           [channel.balance_history_node_0_times, channel.balance_history_node_0_values], \
           [sacrificed_0, sacrificed_1], \
           all_transactions_list
           #[all_transactions_list_node_0, all_transactions_list_node_1]


# if __name__ == '__main__':
#     sc_DES_with_buffer_fun()
