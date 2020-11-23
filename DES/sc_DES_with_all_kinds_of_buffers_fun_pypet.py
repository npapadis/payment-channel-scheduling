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

Max allowed delay is defined as a transaction property, not a channel property.
The buffer is processed every 1 second until the total simulation time has been reached.

To-do (?): Terminate simulation at maximum simulation time, not when a certain number of transactions is generated,
to avoid rejecting the remaining transactions in the buffers, which could have been successful if more transactions were arriving.
"""

from numpy import random
import simpy
import sys
import powerlaw
import pandas as pd


class Transaction:
    def __init__(self, env, channel, time, from_node, to_node, amount, max_allowed_delay, verbose):
        self.env = env
        self.channel = channel
        self.time = time
        self.from_node = from_node
        self.to_node = to_node
        self.amount = amount
        self.max_allowed_delay = max_allowed_delay
        self.verbose = verbose
        self.buffered = False
        self.status = "PENDING"     # Other statuses: "SUCCEEDED", "REJECTED", "EXPIRED"
        self.initially_feasible = None

        if self.verbose:
            print("Transaction {} generated at time {:.2f}.".format(self, self.env.now))

        # Start the run process every time an instance is created.
        # env.process(self.run())

    def run(self):
        if self.status == "PENDING":
            with self.channel.channel_link.request() as request:    # Generate a request event
                yield request                                       # Wait for access to the channel
                self.channel.process_transaction(self)              # Once the channel belongs to the transaction, try to process it.

    def __repr__(self):
        return "%d->%d t=%.2f D=%d a=%d" % (self.from_node, self.to_node, self.time, self.max_allowed_delay, self.amount)


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

    def process_transaction(self, t):

        IP = self.immediate_processing is True          # Immediate Processing
        BE = self.buffers[t.from_node] is not None      # Buffer Exists
        FT = t.buffered is False                        # First Time
        FE = t.amount <= self.balances[t.from_node]     # FEasible
        # Configurations "not BE and not FT" are not reachable. The remaining 12 of the 16 configurations are covered below.

        if t.status != "PENDING":
            print("Error in process_transaction(): attempt to process non-pending transaction (of status \"{}\").".format(t.status))
            sys.exit(1)

        if FT and FE:
            t.initially_feasible = True
        else:
            t.initially_feasible = False

        if (IP and BE and FT and FE) or (IP and BE and not FT and FE) or (IP and not BE and FT and FE) or (
                not IP and BE and not FT and FE):  # process
                # Once the channel belongs to the transaction, then if the deadline has not expired, try to process it.
                if t.time + t.max_allowed_delay >= t.env.now:
                    self.balances[t.from_node] -= t.amount
                    self.balances[t.to_node] += t.amount
                    self.successful_transactions[t.from_node] += 1
                    self.successful_amounts[t.from_node] += t.amount
                    self.balance_history_node_0_times.append(self.env.now)
                    self.balance_history_node_0_values.append(self.balances[0])
                    if FT and self.verbose:
                        print("Transaction {} processed at time {:.2f}.".format(t, self.env.now))
                        print("New balances are", self.balances)
                    t.status = "SUCCEEDED"
                    return True
                else:   # Transaction expired and will be handled in the next processing of the buffer.
                    return False
        elif (IP and BE and FT and not FE) or (not IP and BE and FT):  # add to buffer
            self.buffers[t.from_node].transaction_list.append(t)
            # print(self.buffers[t.from_node].transaction_list)
            t.buffered = True
            if self.verbose:
                print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
                print("Unchanged balances are", self.balances)
                if self.buffers[0] is not None: print("Buffer 0:", self.buffers[0].transaction_list)
                if self.buffers[1] is not None: print("Buffer 1:", self.buffers[1].transaction_list)
            # t.status = "PENDING"  # t.status is "PENDING" already
            return False
        elif (IP and not BE and FT and not FE) or (not IP and not BE and FT):  # reject
            if self.verbose:
                print("Transaction {} rejected at time {:.2f}.".format(t, self.env.now))
                print("Unchanged balances are", self.balances)
            t.status = "REJECTED"
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
        # self.max_allowed_delay = max_allowed_delay
        self.transaction_list = []
        self.processing_order = processing_order
        self.verbose = verbose
        self.total_simulation_time_estimation = total_simulation_time_estimation
        self.total_successes = 0

    def run(self):
        while self.env.now <= self.total_simulation_time_estimation:
            # s = self.process_buffer()
            s = self.process_buffer_greedy(self.processing_order)
            self.total_successes = self.total_successes + s
            yield self.env.timeout(1)

    # def process_buffer(self):    # returns total successful transactions
    #     total_successes_this_time = 0
    #     some_transaction_was_processed_with_current_buffer = 1
    #
    #     while some_transaction_was_processed_with_current_buffer == 1:
    #         some_transaction_was_processed_with_current_buffer = 0
    #         for t in self.transaction_list:
    #             if t.time + t.max_allowed_delay < self.env.now:  # if t is too old, drop it completely
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
    #         #     current_time += max_allowed_delay / 10
    #         # if self.verbose:
    #         #     print("current time: ", current_time)
    #
    #     return total_successes_this_time

    def process_buffer_greedy(self, processing_order):

        if processing_order == "oldest_transaction_first":      # This is FIFO
            sorted_buffer = sorted(self.transaction_list, key=lambda tr: tr.time)
        elif processing_order == "youngest_transaction_first":  # This is LIFO
            sorted_buffer = sorted(self.transaction_list, key=lambda tr: tr.time, reverse=True)
        elif processing_order == "closest_deadline_first":      # This is pointless if max_allowed_delay is the same for all transactions
            sorted_buffer = sorted(self.transaction_list, key=lambda tr: tr.time + tr.max_allowed_delay - self.env.now)
        elif processing_order == "largest_amount_first":
            sorted_buffer = sorted(self.transaction_list, key=lambda tr: tr.amount, reverse=True)
        elif processing_order == "smallest_amount_first":
            sorted_buffer = sorted(self.transaction_list, key=lambda tr: tr.amount)
        else:
            print("Input error: {} is not a valid 'processing_order' value.".format(processing_order))
            sys.exit(1)

        # print("\nBuffer sorted:", sorted_buffer, "\n")

        # Processes all transactions that are possible now and returns total successful transactions.
        total_successes_this_time = 0

        while sorted_buffer:  # while list not empty
            t = sorted_buffer.pop(0)
            if t.time + t.max_allowed_delay < self.env.now:  # if t is too old, reject it and remove it from buffer
                t.status = "EXPIRED"
                self.transaction_list.remove(t)
                if self.verbose:
                    print("FAILURE: Transaction {} expired and was removed from buffer at time {:.2f}.".format(t,
                                                                                                               self.env.now))
                    if self.channel.buffers[0] is not None: print("Buffer 0:", self.channel.buffers[0].transaction_list)
                    if self.channel.buffers[1] is not None: print("Buffer 1:", self.channel.buffers[1].transaction_list)
            else:  # if t is not too old and can be processed, process it
                # if self.channel.process_transaction(t):
                self.env.process(t.run())
                if t.status == "SUCCEEDED":
                    self.transaction_list.remove(t)
                    if self.verbose:
                        print("SUCCESS: Transaction {} was processed and removed from buffer at time {:.2f}.".format(t,
                                                                                                                     self.env.now))
                        print("New balances are", self.channel.balances)
                        if self.channel.buffers[0] is not None: print("Buffer 0:",
                                                                      self.channel.buffers[0].transaction_list)
                        if self.channel.buffers[1] is not None: print("Buffer 1:",
                                                                      self.channel.buffers[1].transaction_list)
                    total_successes_this_time += 1
                else:
                    pass

        return total_successes_this_time


def transaction_generator(env, channel, from_node, total_transactions, max_transaction_amount, exp_mean,
                          max_allowed_delay, all_transactions_list, verbose):
    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        to_node = 1 if (from_node == 0) else 0
        # Amount ~ Uniform
        # amount = random.randint(1, max_transaction_amount)

        # Amount ~ Gaussian
        amount = round(max(1, min(max_transaction_amount, random.normal(max_transaction_amount / 2, max_transaction_amount / 6))))

        # Amount ~ Pareto
        # lower = 1  # the lower end of the support
        # shape = 1.16  # the distribution shape parameter, also known as `a` or `alpha`
        # size = 1  # the size of your sample (number of random values)
        # amount = random.pareto(shape, size) + lower

        # Amount ~ Power-law
        # powerlaw.Power_Law(xmin=1, xmax=2, discrete=True, parameters=[1.16]).generate_random(n=10)

        # Distribution for max_allowed_delay
        # if max_allowed_delay > 0:
        #     delay = random.randint(0, max_allowed_delay)
        # else:
        #     delay = 0
        # t = Transaction(env, channel, env.now, from_node, to_node, amount, delay, verbose)
        t = Transaction(env, channel, env.now, from_node, to_node, amount, max_allowed_delay, verbose)
        all_transactions_list.append(t)

        env.process(t.run())
        # env.process(transaction(env, env.now, from_node, to_node, amount))
        # env.process(transaction(env, channel, env.now, from_node, to_node, amount))
        # print(Transaction(env, env.now, from_node, to_node, amount))
        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)



def sc_DES_with_all_kinds_of_buffers_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0,
                                         total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay,
                                         who_has_buffer, immediate_processing, processing_order, verbose, seed):
    total_simulation_time_estimation = 2 * max(total_transactions_0 * 1 / exp_mean_0,
                                               total_transactions_1 * 1 / exp_mean_1)
    random.seed(seed)

    env = simpy.Environment()

    channel = Channel(env, 0, 1, sum(initial_balances), initial_balances, who_has_buffer, immediate_processing, processing_order, verbose,
                      total_simulation_time_estimation)

    all_transactions_list = []
    env.process(transaction_generator(env, channel, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0,
                                      max_allowed_delay, all_transactions_list, verbose))
    env.process(transaction_generator(env, channel, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1,
                                      max_allowed_delay, all_transactions_list, verbose))
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

        if channel.buffers[0] is not None: print("Buffer 0:", channel.buffers[0].transaction_list)
        if channel.buffers[1] is not None: print("Buffer 1:", channel.buffers[1].transaction_list)

    for t in all_transactions_list:
        del t.env
        del t.channel
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
