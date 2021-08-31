"""
#Main framework that coordinates task assignment and truth inference to deal with different worker activities for defending against data poisoning attack.
#The statistics about Accuracy, Cost, Completion and running time are saved in a "result.txt" file.
"""
import math
import os
import datetime
import random
import traceback

from attacker import Attacker
from extended_td import ExtendedTD
from probabilistic_ta import ProbabilisticTA
from task import Task
from worker import Worker


# noinspection SpellCheckingInspection
class Tdadp:

    # noinspection PyPep8Naming
    def __init__(self, B, alpha, tau, delta):
        # TDADP parameters
        self.B = B  # condition for terminating a batch
        self.alpha = alpha  # probability to assign a golden task to a new worker
        self.tau = tau  #Trust score threshold 
        self.delta = delta  # reliability threshold for marking reliable workers

        # dataset parameters
        self.order = []  # requesting order of workers;
        self.L = 0  # label size
        self.K = 0  # number of workers per task

        # attack parameters
        self.epsilon = 0
        self.lambd = 0

        # ID mapping
        self.id_to_task = {}  # ID to normal task mapping
        self.id_to_worker = {}  # ID to worker mapping
        self.id_to_golden = {}  # ID to golden task mapping
        self.id_to_attacker = {}  # ID to attacker mapping

        # evaluation parameters
        self.a_accuracy = 0  # aggregation accuracy
        self.e_number = 0  # average number of exposed golden tasks
        self.t_cost = 0  # average number of golden task assignment for testing each worker
        self.running_time = 0  # running time of TDADP in millisecond      
           
    def read_normal(self, dataset):
        """ read worker labels on normal tasks """
        try:
            file_path = os.sep.join([dataset, 'input.txt'])
            with open(file_path) as file:
                line = file.readline()
                elements = line.split('\t')
                self.L = int(elements[2])
                self.K = int(elements[3])
                line = file.readline()
                while line:
                    elements = line.split('\t')
                    task_id = int(elements[0])
                    true_label = int(elements[1])
                    worker_num = int(elements[2])
                    task = Task(task_id, true_label, self.L)
                    self.id_to_task[task_id] = task
                    for i in range(0, worker_num):
                        worker_id = int(elements[2 * i + 3])
                        answer = int(elements[2 * i + 4])
                        if worker_id not in self.id_to_worker.keys():
                            worker = Worker()
                            self.id_to_worker[worker_id] = worker
                            worker.add_pair(task, answer)
                            task.add_worker(worker)
                        else:
                            worker = self.id_to_worker[worker_id]
                            worker.add_pair(task, answer)
                            task.add_worker(worker)
                    line = file.readline()
        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def read_golden(self, dataset):
        try:
            """ read worker labels on golden tasks """
            file_path = os.sep.join([dataset, 'golden.txt'])
            with open(file_path) as file:
                line = file.readline()
                golden_num = int(line)
                line = file.readline()
                elements = line.split('\t')
                for i in range(0, golden_num):
                    golden_id = int(elements[i * 2])
                    task = Task(golden_id, int(elements[i * 2 + 1]), self.L)
                    self.id_to_golden[golden_id] = task
                line = file.readline()
                while line:
                    elements = line.split('\t')
                    worker_id = int(elements[0])
                    worker = self.id_to_worker[worker_id]
                    for j in range(0, golden_num):
                        golden_id = int(elements[j * 2 + 1])
                        task = self.id_to_golden[golden_id]
                        answer = int(elements[j * 2 + 2])
                        worker.add_pair(task, answer)
                    line = file.readline()
        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def read_attack(self, dataset, r):
        """ read malicious workers of each attacker for the rth run """
        try:
            self.id_to_attacker = {}
            file_path = os.sep.join([dataset, str(r), 'attack.txt'])
            with open(file_path) as file:
                line = file.readline()
                elements = line.split('\t')
                self.epsilon = float(elements[1])
                self.lambd = int(elements[2])
                for i in range(0, self.lambd):
                    line = file.readline()
                    elements = line.split('\t')
                    attacker_id = int(elements[0])
                    task_num = int(elements[1])
                    attacker = Attacker(self.K, self.L)
                    self.id_to_attacker[attacker_id] = attacker
                    for j in range(0, task_num):
                        task_id = int(elements[2 * j + 2])
                        label = int(elements[2 * j + 3])
                        if task_id in self.id_to_task.keys():
                            attacker.set_task_label(self.id_to_task[task_id], label)
                        else:
                            attacker.set_task_label(self.id_to_golden[task_id], label)
                    line = file.readline()
                    elements = line.split('\t')
                    attacker_id = int(elements[0])
                    worker_num = int(elements[1])
                    for j in range(0, worker_num):
                        worker_id = int(elements[j + 2])
                        worker = self.id_to_worker[worker_id]
                        worker.set_attacker_id(attacker_id)
        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def read_order(self, dataset, r):
        """ read requesting order of workers for the rth run """
        try:
            self.order = []
            file_path = os.sep.join([dataset, str(r), 'order.txt'])
            with open(file_path) as file:
                line = file.readline()
                while line:
                    worker_id = int(line)
                    self.order.append(self.id_to_worker[worker_id])
                    line = file.readline()
        except Exception as e:
            print(e)
            traceback.print_stack()
            traceback.print_exc()

    def run(self):
        workers = set()  # current workers in U
        promotion_num = 0  # number of completed tasks that can be promoted
        gold_num = 0  # number of golden task assignment
        etd = ExtendedTD(self.L)  # truth inference
        pta = ProbabilisticTA(self.tau, self.delta, self.alpha, self.K)  #Task assignment

        start_time = datetime.datetime.now().timestamp()
        # respond to different worker activity
        for worker in self.order:
            # case 1: a worker requests
            if worker.is_banned():
                continue
            else:
                workers.add(worker)
            golden_tasks = set(self.id_to_golden.values())
            normal_tasks = set(self.id_to_task.values())
            assigned_task = pta.assign(worker, golden_tasks, normal_tasks)
            if assigned_task and worker.get_attacker_id() != -1:
                attacker = self.id_to_attacker[worker.get_attacker_id()]
                # update the observation of the attacker if a task is assigned to a malicious worker
                attacker.observe(assigned_task)

            # case 2: a worker labels a golden task
            if assigned_task in self.id_to_golden.values():
                gold_num += 1
                if worker.get_attacker_id() == -1:
                    worker.label(assigned_task, worker.get_pairs()[assigned_task])
                else:
                    attacker = self.id_to_attacker[worker.get_attacker_id()]
                    label = attacker.get_task_label(assigned_task)
                    worker.label(assigned_task, label)

                # update s_j, r_j and p_j
                s_count = 0
                r_count = 0
                r_correct = 0
                for task in worker.get_labeled_pairs().keys():
                    if task in self.id_to_golden.values():
                        r_count += 1
                        task.cal_majority()
                        majority = task.get_majority()
                        truth = task.get_true_label()
                        if task in self.id_to_task.values():
                            truth = task.get_aggregated()
                        answer = worker.get_labeled_pairs()[task]
                        for i in range(0, self.L):
                            if majority[i] == 1 and answer == i and i != truth:
                                s_count += 1
                        if answer == truth:
                            r_correct += 1
                worker.set_s((2.0 / (1 + math.pow(math.e, -s_count))) - 1)
               
            # "worker.setR((2.0/(1+Math.pow(Math.E, -r_count/3))-1)*r_correct/r_count);
                worker.set_r((2.0 / (1 + math.pow(math.e, -r_count // 3)) - 1) * r_correct // r_count)
                              # "worker.setP(r_correct/r_count);
                worker.set_p(r_correct // r_count)

                # ban the worker if trust score passes the threshold
                if worker.get_s() >= self.tau:
                    worker.ban()
                    # remove the worker's labels on normal tasks
                    workers.remove(worker)
                    to_remove = set(worker.get_labeled_pairs().keys())
                    to_remove.difference_update(self.id_to_golden.values())
                    for task in to_remove:
                        task.expose()
                        worker.remove(task)
                        task.remove(worker)

            # case 3: a worker labels a normal task
            elif assigned_task in self.id_to_task.values():
                attacker_id = worker.get_attacker_id()
                if attacker_id != -1:
                    label = self.id_to_attacker[attacker_id].get_task_label(assigned_task)

                    # occasionally deviate from the sharing
                    rand = random.Random()
                    if rand.random() <= self.epsilon:
                        temp_label = rand.randint(0, self.L - 1)
                        while temp_label == label:
                            temp_label = random.randint(0, self.L - 1)
                        label = temp_label
                else:
                    label = worker.get_pairs().get(assigned_task)
                worker.label(assigned_task, label)

                # update the number of completed tasks that can be promoted
                if len(assigned_task.get_assigned()) >= self.K:
                    assigned_task.calc_ci()
                    if assigned_task.get_ci() >= self.delta:
                        promotion_num += 1

            # if the batch condition is met, update aggregated labels and promote tasks
            if promotion_num == self.B:
                tasks = set(self.id_to_task.values())
                tasks.difference_update(self.id_to_golden.values())
                # run truth inference
                etd.process(tasks, workers)
                for task in tasks:
                    if len(task.get_assigned()) >= self.K:
                        task.calc_ci()
                        if task.get_ci() >= self.delta:
                            self.id_to_golden[task.get_task_id()] = task
                promotion_num = 0

        tasks = set(self.id_to_task.values())
        etd.process(tasks, workers)
        end_time = datetime.datetime.now().timestamp()

        self.a_accuracy = 0
        for task in self.id_to_task.values():
            if task.get_aggregated() == task.get_true_label():
                self.a_accuracy += 1
            task.reset()
        for worker in self.id_to_worker.values():
            worker.reset()
        self.a_accuracy = self.a_accuracy / len(self.id_to_task)
        exposed = set()
        for attacker in self.id_to_attacker.values():
            for golden in self.id_to_golden.values():
                if attacker.get_count(golden) > self.K:
                    exposed.add(golden)
        self.e_number = len(exposed)
        self.t_cost = gold_num / len(self.id_to_worker)
        self.running_time = (end_time - start_time) * 1000

    def get_a_accuracy(self):
        """ return the aggregation accuracy """
        return self.a_accuracy

    def get_e_number(self):
        """ return the average number of exposed golden tasks """
        return self.e_number

    def get_t_cost(self):
        """ return the average number of golden task assignment for testing each worker """
        return self.t_cost

    def get_running_time(self):
        """ return the running time """
        return self.running_time
