#Modeling of tasks.
#Each task is associated with an average reliability of workers on the task and a true label. A task is completed once the
#aggregated answers are determined.


class Task:

    # noinspection PyPep8Naming
    def __init__(self, task_id, true_label, L):
        self.workers = []  # assigned workers in the original data
        self.assigned = []  # current assigned workers
        self.true_label = true_label  # true label of the task
        self.aggregated = -1  # aggregated label of the task
        self.majority = [0] * L  # majority indicator (1 means that the corresponding label is a majority vote)
        self.L = L  # label size
        self.task_id = task_id  # task ID
        self.c_i = 0  # average reliability score of assigned workers
        self.exposed = 0  # number of times assigned to banned workers

    def get_task_id(self):
        """ return task ID """
        return self.task_id

    def add_worker(self, worker):
        """ update the original assigned workers for the task """
        self.workers.append(worker)

    def get_workers(self):
        """ return the original assigned workers for the task """
        return self.workers

    def assign(self, worker):
        """" assign a worker to the task """
        self.assigned.append(worker)

    def get_assigned(self):
        """ return assigned workers """
        return self.assigned

    def remove(self, worker):
        """ remove a banned worker from assigned workers for a normal task """
        self.assigned.remove(worker)

    def get_true_label(self):
        """ return the true label """
        return self.true_label

    def set_aggregated(self, aggregated):
        """ set the aggregated label """
        self.aggregated = aggregated

    def get_aggregated(self):
        """ return the aggregated label """
        return self.aggregated

    def cal_majority(self):
        """ compute the indicator of majority labels that receive the most votes """
        max_vote = 0
        for i in range(0, self.L):
            self.majority[i] = 0
        for worker in self.assigned:
            answer = worker.get_labeled_pairs().get(self)
            self.majority[answer] = self.majority[answer] + 1
            if self.majority[answer] > max_vote:
                max_vote = self.majority[answer]
        for j in range(0, self.L):
            if self.majority[j] == max_vote and max_vote >= 2:
                self.majority[j] = 1
            else:
                self.majority[j] = 0

    def get_majority(self):
        """ return the indicator of majority labels that receive the most votes """
        return self.majority

    def expose(self):
        """ update the number of times being assigned to banned workers """
        self.exposed += + 1

    def get_expose(self):
        """ return the number of times being assigned to banned workers """
        return self.exposed

    def calc_ci(self):
        """ compute the average reliability of assigned workers """
        self.c_i = 0
        for worker in self.assigned:
            self.c_i += worker.get_r()
        self.c_i = self.c_i / len(self.assigned)

    def get_ci(self):
        """ return the average reliability of assigned workers """
        return self.c_i

    def reset(self):
        """ reset the features of the task for a new run """
        self.assigned = []
        self.aggregated = -1
        self.c_i = 0
        self.exposed = 0
