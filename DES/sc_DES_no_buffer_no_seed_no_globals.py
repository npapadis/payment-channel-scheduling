"""
This script uses Discrete Event Simulation via SimPy to simulate a single payment channel.
Transactions are generated from both sides according to customizable distributions of amounts and arrival times.
A transaction arriving at the channel is either processed immediately if possible, or rejected immediately (no buffer).
"""



from numpy import random
import simpy


class Transaction:
    def __init__(self, env, channel, time, from_node, to_node, amount):
        self.env = env
        self.channel = channel
        self.time = time
        self.from_node = from_node
        self.to_node = to_node
        self.amount = amount
        # Start the run process every time an instance is created.
        # env.process(self.run())

    def run(self):
        if verbose:
            print("Transaction", self, "generated at time %.2f." % self.env.now)

        # If it can be processed now, do so
        with self.channel.channel_link.request() as req:  # Generate a request event
            yield req
            self.channel.process_transaction(self)

        # if it can't be processed now, queue in buffer
        # yield self.env.timeout(1)

    def __repr__(self):
        return "%d->%d t=%.2f a=%d" % (self.from_node, self.to_node, self.time, self.amount)

# def transaction(env, time, from_node, to_node, amount):
# # def transaction(env, channel, time, from_node, to_node, amount):
#
#     # while True:
#     # if it can be processed now, do so
#     # if it can't be processed now, queue in buffer
#     print("hi at time ",time)
#     yield env.timeout(1)


class Channel:

    # def __init__(self, env, node1, node2, capacity, balances):
    def __init__(self, env, node1, node2, capacity, balances):
        self.env = env
        self.node1 = node1
        self.node2 = node2
        self.capacity = capacity
        self.balances = balances
        self.channel_link = simpy.Resource(env, capacity=1)
        self.successful_transactions = [0, 0]
        self.successful_amounts = [0, 0]

    def process_transaction(self, t):
        if t.amount <= self.balances[t.from_node]:
            self.balances[t.from_node] -= t.amount
            self.balances[t.to_node] += t.amount
            self.successful_transactions[t.from_node] += 1
            self.successful_amounts[t.from_node] += t.amount
            if verbose:
                print("Transaction", t, "processed at time %.2f" % self.env.now, ".")
                print("New balances are", self.balances)
        else:  # Add to buffer
            if verbose:
                print("Transaction", t, "added to buffer of node", t.from_node, ".")
                print("Unchanged balances are", self.balances)



# def transaction_generator(env, from_node, total_transactions, max_transaction_amount, exp_mean):
def transaction_generator(env, channel, from_node, total_transactions, max_transaction_amount, exp_mean):

    # def __init__(self, env, from_node, total_transactions, max_transaction_amount, exp_mean):
    #     self.env = env
    #     self.from_node = from_node
    #     self.total_transactions = total_transactions
    #     self.max_transaction_amount = max_transaction_amount
    #     self.exp_mean = exp_mean

    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        to_node = 1 if (from_node == 0) else 0
        amount = random.randint(1, max_transaction_amount)
        t = Transaction(env, channel, env.now, from_node, to_node, amount)
        env.process(t.run())
        # env.process(transaction(env, env.now, from_node, to_node, amount))
        # env.process(transaction(env, channel, env.now, from_node, to_node, amount))
        # print(Transaction(env, env.now, from_node, to_node, amount))
        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)


verbose = True


def main():
    initial_balances = [70, 30]
    # total_simulation_time = 1000.0


    # Node 0:
    total_transactions_0 = 10
    max_transaction_amount_0 = sum(initial_balances)
    exp_mean_0 = 1 / 3

    # Node 1:
    total_transactions_1 = 10
    max_transaction_amount_1 = sum(initial_balances)
    exp_mean_1 = 1 / 5


    env = simpy.Environment()

    channel = Channel(env, 0, 1, sum(initial_balances), initial_balances)
    env.process(transaction_generator(env, channel, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0))
    env.process(transaction_generator(env, channel, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1))
    # env.process(transaction_generator(env, 0, total_transactions_0, max_transaction_amount_0, exp_mean_0))
    # env.process(transaction_generator(env, 1, total_transactions_1, max_transaction_amount_1, exp_mean_1))

    env.run()

    success_rate = [channel.successful_transactions[0] / total_transactions_0,
                    channel.successful_transactions[1] / total_transactions_1]
    if verbose:
        print("Success rate:", success_rate)
        print("Total successfully processed amounts:", channel.successful_amounts)


if __name__ == '__main__':
    main()
