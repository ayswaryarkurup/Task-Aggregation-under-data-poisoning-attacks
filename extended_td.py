
 #Provides the truth inference, which iteratively infers the true answer of tasks and the quality of workers  with the trust score and reliability score of workers.


class ExtendedTD:

    # noinspection PyPep8Naming
    def __init__(self, L):
        self.L = L

    def process(self, tasks, workers):
        # iteratively run truth inference
        # set the initial weight of workers to their accuracy on golden tasks
        for worker in workers:
            worker.set_weight(worker.get_p())

        iteration = 0
        while iteration < 1000:
            iteration += 1
            difference = 0

            # task aggregation
            for task in tasks:
                original_label = task.get_aggregated()
                assigned = task.get_assigned()
                votes = [0] * self.L
                for worker in assigned:
                    label = worker.get_labeled_pairs()[task]
                    votes[label] = votes[label] + worker.get_s() / self.L + (1 - worker.get_s()) * worker.get_weight()
                max_vote = -1
                for i in range(0, self.L):
                    if votes[i] > max_vote:
                        task.set_aggregated(i)
                        max_vote = votes[i]
                if original_label != task.get_aggregated():
                    difference += 1

            # terminate if converge
            if difference == 0:
                break

            # weight estimation
            for worker in workers:
                correct = 0
                count = 0
                assigned = worker.get_labeled_pairs()
                for task in assigned.keys():
                    if assigned[task] == task.get_aggregated():
                        correct += task.get_ci()
                    count += task.get_ci()
                if count > 0:
                    worker.set_weight(correct / count)
