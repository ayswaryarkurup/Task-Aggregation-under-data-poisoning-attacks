
  #Formalizing a real dataset or generate a synthetic  dataset.,
#An "input.txt" file and a "golden.txt" file are created to indicate  the (task, worker, label) triples for the tasks. For  each run, an "attacker.txt" file and an "order.txt" file are also created to  simulate different replacements of independent workers with malicious workers and  different order of worker requests, respectively.


import os
import random
import math
import traceback
from pathlib import Path


class Preprocess:

    # noinspection PyPep8Naming
    def __init__(self, dataset, run_num, mu, epsilon, lamb, N=0, M=0, L=0, K=0, theta=0):
        """
        Initialization for real datasets  when used without default parameters.
         Initialization for synthetic datasets otherwise
        """
        self.dataset = dataset
        self.run_num = run_num  # number of test runs
        self.golden_num = 20  # number of golden tasks (20 by default)
        self.mu = mu  # percentage of malicious workers
        self.epsilon = epsilon  # probability for malicious workers to deviate from sharing
        self.lamb = lamb  # number of attackers
        self.N = N  # number of tasks
        self.M = M  # number of workers
        self.L = L  # optional label size
        self.K = K  # number of workers per normal task
        self.theta = theta  # average worker accuracy
        self.has_golden = False  # indicates whether golden tasks are provided

        self.worker_normal_labels = {}  # (task, label) pairs of each worker on normal tasks
        self.normal_workers = {}  # workers on each normal task
        self.normal_truth = {}  # true label of each normal task
        self.worker_golden_labels = {}  # (golden task, label) pairs of each worker on golden tasks
        self.golden_truth = {}  # true label of each golden task
        self.attacker_sybils = {}  # malicious workers of each attacker
        self.worker_id = {}  # worker to id mapping

    @classmethod
    def real_dataset(cls, dataset, run_num, mu, epsilon, lamb):
        """ Initialization for real datasets (NLP and DOG) """
        return cls(dataset, run_num, mu, epsilon, lamb)

    # noinspection PyPep8Naming
    @classmethod
    def synth_dataset(cls, dataset, run_num, mu, epsilon, lamb, N, M, L, K, theta):
        """ Initialization for synthetic datasets (SYN) """
        return cls(dataset, run_num, mu, epsilon, lamb, N, M, L, K, theta)

    def read_data(self):
        """
        read (task, worker, label) tuples of normal tasks in answer.csv,
        read (task, true label) pairs of normal tasks in truth.csv,
        read (task, worker, label) tuples of golden tasks in quali.csv
        and read (task, true label) pairs of golden tasks in quali_truth.csv
        """
        # noinspection PyBroadException
        try:
            with open(self._file_path('answer.csv')) as answer_file:
                answer_file.readline()
                answer_line = answer_file.readline()
                _id = 0
                self.L = 0
                self.K = 0
                while answer_line:
                    elements = answer_line.split(",")
                    task = int(elements[0])
                    # map string ID to integer ID
                    worker = _id
                    if elements[1] in self.worker_id.keys():
                        worker = self.worker_id[elements[1]]
                    else:
                        self.worker_id[elements[1]] = worker
                        _id += 1
                    label = int(elements[2])
                    if label + 1 > self.L:
                        self.L = label + 1

                    # update worker_normal_labels
                    normal_labels = self.worker_normal_labels.get(worker)
                    if not normal_labels:
                        normal_labels = {}
                    normal_labels[task] = label
                    self.worker_normal_labels[worker] = normal_labels

                    # update normal_workers
                    workers = self.normal_workers.get(task)
                    if not workers:
                        workers = []
                    workers.append(worker)
                    if len(workers) > self.K:
                        self.K = len(workers)
                    self.normal_workers[task] = workers

                    answer_line = answer_file.readline()
                self.N = len(self.normal_workers)
                self.M = len(self.worker_id)

            with open(self._file_path('truth.csv')) as truth_file:
                truth_file.readline()
                truth_line = truth_file.readline()
                while truth_line:
                    elements = truth_line.split(",")
                    task = int(elements[0])
                    truth = int(elements[1])

                    # update normal_truth
                    self.normal_truth[task] = truth

                    truth_line = truth_file.readline()

            if os.path.exists(self._file_path('quali.csv')):
                self.has_golden = True
            if self.has_golden:
                with open(self._file_path('quali.csv')) as quali_file:
                    quali_file.readline()
                    quali_line = quali_file.readline()
                    while quali_line:
                        elements = quali_line.split(",")
                        task = int(elements[0])
                        worker = self.worker_id[elements[1]]
                        label = int(elements[2])

                        # update worker_golden_labels
                        golden_labels = self.worker_golden_labels.get(worker)
                        if not golden_labels:
                            golden_labels = {}
                        golden_labels[task] = label
                        self.worker_golden_labels[worker] = golden_labels
                        quali_line = quali_file.readline()

                with open(self._file_path('quali_truth.csv')) as quali_truth_file:
                    quali_truth_file.readline()
                    quali_truth_line = quali_truth_file.readline()
                    while quali_truth_line:
                        elements = quali_truth_line.split(",")
                        task = int(elements[0])
                        truth = int(elements[1])

                        # update golden_truth
                        self.golden_truth[task] = truth

                        quali_truth_line = quali_truth_file.readline()

                self.golden_num = len(self.golden_truth)

        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def simulate(self):
        # create tasks and their true label
        rand = random.Random()
        for i in range(0, self.N):
            self.normal_truth[i] = rand.randint(0, self.L - 1)

        # create workers and their label on tasks
        avail_workers = []
        for j in range(0, self.M):
            avail_workers.append(j)
            self.worker_normal_labels[j] = {}
        answer_num = self.N * self.K / self.M + 5
        removed = []
        for k in range(0, self.N):
            random.shuffle(avail_workers)
            t_workers = []
            for t in range(0, self.K):
                worker = avail_workers[len(avail_workers) - 1 - t]
                task_labels = self.worker_normal_labels[worker]
                truth = self.normal_truth[k]
                if rand.random() <= self.theta:
                    task_labels[k] = truth
                else:
                    label = rand.randint(0, self.L - 1)
                    while label == truth:
                        label = rand.randint(0, self.L - 1)
                    task_labels[k] = label
                self.worker_normal_labels[worker] = task_labels
                if len(task_labels) == answer_num:
                    removed.append(worker)
                t_workers.append(worker)
            avail_workers = []
            for p in range(0, self.M):
                if p not in removed:
                    avail_workers.append(p)
            self.normal_workers[k] = t_workers

    def replace(self):
        """
        replace mu percentage of independent normal workers with malicious workers
        and equally assign malicious workers to lambda attackers
        """
        self.attacker_sybils = {}
        # decide the number of malicious workers for each attacker
        num = math.ceil(len(self.worker_normal_labels) * self.mu)
        attacker_sybil_num = [0] * self.lamb
        for i in range(0, self.lamb):
            # "(int) Math.floor(num/lambda)" 
            attacker_sybil_num[i] = num // self.lamb
        attacker_sybil_num[self.lamb - 1] = num - attacker_sybil_num[0] * (self.lamb - 1)

        # update attacker_trust score by randomly assigning independent workers to each attacker as malicious workers
        temp_workers = list(self.worker_normal_labels.keys())
        random.shuffle(temp_workers)
        for j in range(0, self.lamb):
            workers = []
            for k in range(0, attacker_sybil_num[j]):
                worker = temp_workers.pop(len(temp_workers) - 1)
                workers.append(worker)
            self.attacker_sybils[j] = workers

    def formalize(self):
        """
        write the overall data info into input.txt and write the information
        of data poisoning attack, golden tasks and request order for each run
        """
        rand = random.Random()
        try:
            with open(self._file_path('input.txt'), 'w') as input_file:
                # write worker number M, task number N, label size L and worker number per task K
                input_file.write(f'{self.M}\t{self.N}\t{self.L}\t{self.K}\n')
                for task in self.normal_truth.keys():
                    # write task ID, true label and number of workers for each task
                    input_file.write(f'{task}\t{self.normal_truth[task]}\t{len(self.normal_workers[task])}\t')
                    # write worker ID and corresponding label
                    for worker in self.normal_workers[task]:
                        input_file.write(f'{worker}\t{self.worker_normal_labels[worker][task]}\t')
                    input_file.write('\n')

            with open(self._file_path('golden.txt'), 'w') as golden_file:
                golden_file.write(f'{self.golden_num}\n')
                if self.has_golden:
                    # write the true label of each golden task
                    for golden in self.golden_truth.keys():
                        golden_file.write(f'{golden}\t{self.golden_truth[golden]}\t')
                    golden_file.write('\n')
                    # write the label of each worker on each golden task
                    for worker in self.worker_golden_labels.keys():
                        golden_file.write(f'{worker}\t')
                        golden_labels = self.worker_golden_labels[worker]
                        for golden in golden_labels.keys():
                            golden_file.write(f'{golden}\t{golden_labels[golden]}\t')
                        golden_file.write('\n')
                else:
                    # generate golden tasks
                    golden_labels = [0] * self.golden_num
                    for i in range(0, self.golden_num):
                        golden_labels[i] = rand.randint(0, self.L - 1)
                        golden_file.write(f'{-1 - i}\t{golden_labels[i]}\t')
                    golden_file.write('\n')
                    # determine the label provided by each worker on golden tasks
                    for worker in self.worker_normal_labels.keys():
                        golden_file.write(f'{worker}\t')
                        # compute the worker's accuracy
                        acc = 0
                        task_labels = self.worker_normal_labels[worker]
                        for task in task_labels.keys():
                            if self.normal_truth[task] == task_labels[task]:
                                acc += 1
                        acc = acc / len(task_labels)

                        # generate the worker's label on each golden task based on the computed accuracy
                        for i in range(0, self.golden_num):
                            golden_file.write(f'{-1 - i}\t')
                            if rand.random() <= acc:
                                golden_file.write(f'{golden_labels[i]}\t')
                            else:
                                answer = rand.randint(0, self.L - 1)
                                while answer == golden_labels[i]:
                                    answer = rand.randint(0, self.L - 1)
                                golden_file.write(f'{answer}\t')
                        golden_file.write('\n')

            for run in range(0, self.run_num):
                dir_path = os.sep.join([self.dataset, str(run)])
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.replace()

                # attack.txt contains the labels randomized by each attacker and the malicious workers controlled by each
                # attacker
                attack_file_path = os.sep.join([self.dataset, str(run), "attack.txt"])
                with open(attack_file_path, 'w') as attack_file:
                    attack_file.write(f'{self.mu}\t{self.epsilon}\t{self.lamb}\n')
                    for i in range(0, self.lamb):
                        # write attacker ID and total number of tasks for each attacker
                        attack_file.write(f'{i}\t{len(self.normal_truth) + self.golden_num}\t')
                        # write task ID and randomized label for normal tasks
                        for task in self.normal_truth.keys():
                            attack_file.write(f'{task}\t{rand.randint(0, self.L - 1)}\t')
                        # write task ID and randomized label for golden tasks
                        if self.has_golden:
                            for golden in self.golden_truth.keys():
                                attack_file.write(f'{golden}\t{rand.randint(0, self.L - 1)}\t')
                        else:
                            for j in range(-1, -self.golden_num - 1, -1):
                                attack_file.write(f'{j}\t{rand.randint(0, self.L - 1)}\t')
                        attack_file.write('\n')
                        workers = self.attacker_sybils.get(i)
                        # write attacker ID and number of malicious workers for each attacker
                        attack_file.write(f'{i}\t{len(workers)}\t')
                        # write worker ID of replaced independent workers
                        for worker in workers:
                            attack_file.write(f'{worker}\t')
                        attack_file.write('\n')

                order_file_path = os.sep.join([self.dataset, str(run), "order.txt"])
                with open(order_file_path, 'w') as order_file:
                    order = []
                    # decide the number of requests for each worker
                    for worker in self.worker_normal_labels.keys():
                        task_labels = self.worker_normal_labels[worker]
                        for i in range(0, len(task_labels)):
                            order.append(worker)
                        for j in range(0, self.golden_num):
                            order.append(worker)
                    # randomize the request order
                    random.shuffle(order)
                    for worker in order:
                        order_file.write(f'{worker}\n')
        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def _file_path(self, file_name):
        return os.sep.join([self.dataset, file_name])
