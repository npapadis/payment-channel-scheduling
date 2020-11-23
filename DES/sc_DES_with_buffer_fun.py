"""
This script uses Discrete Event Simulation via SimPy to simulate a single payment channel.
Transactions are generated from both sides according to customizable distributions of amounts and arrival times.
A transaction arriving at the channel is processed immediately, if possible.
Otherwise, it is added to a buffer until it processed successfully or its deadline expires.
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
        # Start the run process every time an instance is created.
        # env.process(self.run())

    def run(self):
        if self.verbose:
            print("Transaction {} generated at time {:.2f}.".format(self, self.env.now))

        # If it can be processed now, do so
        with self.channel.channel_link.request() as req:  # Generate a request event
            yield req
            # Once the channel belongs to the transaction, then if the deadline has not expired, try to process it.
            if self.time + self.max_allowed_delay >= self.env.now:
                self.channel.process_transaction(self)

    def __repr__(self):
        return "%d->%d t=%.2f a=%d" % (self.from_node, self.to_node, self.time, self.amount)


class Channel:

    def __init__(self, env, node0, node1, capacity, balances, verbose, total_simulation_time_estimation):
        self.env = env
        self.node0 = node0
        self.node1 = node1
        self.capacity = capacity
        self.balances = balances
        self.verbose = verbose
        self.channel_link = simpy.Resource(env, capacity=1)
        self.successful_transactions = [0, 0]
        self.successful_amounts = [0, 0]
        self.balance_history_node_0_times = []
        self.balance_history_node_0_values = []
        self.buffers = [Buffer(env, node0, self, verbose, total_simulation_time_estimation), Buffer(env, node1, self, verbose, total_simulation_time_estimation)]
        self.env.process(self.buffers[0].run())
        self.env.process(self.buffers[1].run())
        # self.buffer1 = Buffer(node1)

    def process_transaction(self, t):
        if t.amount <= self.balances[t.from_node]:
            self.balances[t.from_node] -= t.amount
            self.balances[t.to_node] += t.amount
            self.successful_transactions[t.from_node] += 1
            self.successful_amounts[t.from_node] += t.amount
            self.balance_history_node_0_times.append(self.env.now)
            self.balance_history_node_0_values.append(self.balances[0])
            if self.verbose:
                print("Transaction {} processed at time {:.2f}.".format(t, self.env.now))
                print("New balances are", self.balances)
            return True
        else:  # Add to buffer
            if t.buffered is False:
                self.buffers[t.from_node].transaction_list.append(t)
                # print(self.buffers[t.from_node].transaction_list)
                t.buffered = True
                if self.verbose:
                    print("Transaction {} added to buffer of node {}.".format(t, t.from_node))
                    print("Unchanged balances are", self.balances)
            return False


class Buffer:
    def __init__(self, env, node, channel, verbose, total_simulation_time_estimation):
        self.env = env
        self.node = node
        self.channel = channel
        # self.max_allowed_delay = max_allowed_delay
        self.transaction_list = []
        self.verbose = verbose
        self.total_simulation_time_estimation = total_simulation_time_estimation
        self.total_successes = 0

    def run(self):
        while self.env.now <= self.total_simulation_time_estimation:
            s = self.process_buffer()
            self.total_successes = self.total_successes + s
            yield self.env.timeout(1)


    def process_buffer(self):    # returns total successful transactions
        total_successes_this_time = 0
        some_transaction_was_processed_with_current_buffer = 1

        while some_transaction_was_processed_with_current_buffer == 1:
            some_transaction_was_processed_with_current_buffer = 0
            for t in self.transaction_list:
                if t.time + t.max_allowed_delay < self.env.now:  # if t is too old, drop it completely
                    self.transaction_list.remove(t)
                    if self.verbose:
                        print("FAILURE: Transaction {} expired and was removed from buffer at time {:.2f}.".format(t, self.env.now))

                else:  # if t is not too old and can be processed, process it
                    if self.channel.process_transaction(t):
                        self.transaction_list.remove(t)
                        if self.verbose:
                            print("SUCCESS: Transaction {} was processed and removed from buffer at time {:.2f}.".format(t, self.env.now))
                        some_transaction_was_processed_with_current_buffer = 1
                        total_successes_this_time += 1
                    else:
                        pass
            # if no_more_incoming_transactions:
            #     current_time += max_allowed_delay / 10
            # if self.verbose:
            #     print("current time: ", current_time)

        return total_successes_this_time


def transaction_generator(env, channel, from_node, total_transactions, max_transaction_amount, exp_mean,
                          max_allowed_delay, verbose):
    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        to_node = 1 if (from_node == 0) else 0
        # amount = random.randint(1, max_transaction_amount)
        amount = min(max_transaction_amount, random.normal(max_transaction_amount/2, max_transaction_amount/6))
        # amount = (random.pareto(1, 1) + 1) * 2
        t = Transaction(env, channel, env.now, from_node, to_node, amount, max_allowed_delay, verbose)
        env.process(t.run())
        # env.process(transaction(env, env.now, from_node, to_node, amount))
        # env.process(transaction(env, channel, env.now, from_node, to_node, amount))
        # print(Transaction(env, env.now, from_node, to_node, amount))
        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)


def sc_DES_with_buffer_fun(initial_balances, total_transactions_0, max_transaction_amount_0, exp_mean_0, total_transactions_1, max_transaction_amount_1, exp_mean_1, max_allowed_delay, verbose, seed):

    total_simulation_time_estimation = 2*max(total_transactions_0*1/exp_mean_0, total_transactions_1*1/exp_mean_1)
    random.seed(seed)

    env = simpy.Environment()

    channel = Channel(env, 0, 1, sum(initial_balances), initial_balances, verbose, total_simulation_time_estimation)

    env.process(transaction_generator(env, channel, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0,
                                      max_allowed_delay, verbose))
    env.process(transaction_generator(env, channel, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1,
                                      max_allowed_delay, verbose))
    # env.process(transaction_generator(env, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0))
    # env.process(transaction_generator(env, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1))

    env.run()

    success_rates = [channel.successful_transactions[0] / total_transactions_0,
                     channel.successful_transactions[1] / total_transactions_1,
                     (channel.successful_transactions[0] + channel.successful_transactions[1]) / (total_transactions_0 + total_transactions_1)]

    if verbose:
        print("Success rate:", success_rates)
        print("Total successfully processed amounts:", channel.successful_amounts)

    # print(channel.buffers[0].transaction_list)
    # print(channel.buffers[1].transaction_list)

    return success_rates, channel.successful_amounts, [channel.balance_history_node_0_times, channel.balance_history_node_0_values]

# if __name__ == '__main__':
#     sc_DES_with_buffer_fun()
