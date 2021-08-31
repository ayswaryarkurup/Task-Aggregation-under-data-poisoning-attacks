"""
 Provides the data poisoning attacker. Each attacker is  associated with a set of malicious workers. For each targeted task, the attacker randomizes a label or answer for malicious workers which to be share. For attack detection, it keeps  counting the observation of tasks so that a golden task could identified if it is assigned to more than k malicious workers under his control. If a golden task is identified, the attacker would honestly provide the answer for malicious workers  to share, where the new answer has theta probability to be the true answer.
"""

from random import Random


class Attacker:

    # noinspection PyPep8Naming
    def __init__(self, K, L):
        self.task_label = {}  # randomized label on each task
        self.task_count = {}  # observation times of each task
        self.K = K  # number of workers for each task
        self.L = L  # label size

    def set_task_label(self, task, label):
        """ set a randomized label on a task """
        self.task_label[task] = label

    def get_task_label(self, task):
        """ return the label on a task """
        return self.task_label[task]

    def observe(self, task):
        """ update the observation times of a task """
        if task in self.task_count.keys():
            count = self.task_count[task]
            count += 1
            self.task_count[task] = count
            # label the task honestly if the task is observed for more than K times
            if count == self.K + 1:
                rand = Random()
                if rand.random() <= 0.8:
                    self.set_task_label(task, task.get_true_label())
                else:
                    label = rand.randint(0, self.L - 1)
                    while label == task.get_true_label():
                        label = rand.randint(0, self.L - 1)
                    self.set_task_label(task, label)
        else:
            self.task_count[task] = 1

    def get_count(self, task):
        """ return the observation times of a task """
        if task in self.task_count.keys():
            return self.task_count[task]
        else:
            return 0
