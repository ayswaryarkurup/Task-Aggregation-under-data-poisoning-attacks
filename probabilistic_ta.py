
#Provides the component of task assignment based on the truth inference and worker scores, which assigns a task when a worker requests.

from random import Random


class ProbabilisticTA:

    def __init__(self, tau, delta, alpha, K):
        self.tau = tau  # trust score threshold for banning workers
        self.delta = delta  # reliability threshold for marking reliable workers
        self.alpha = alpha  # probability to assign a golden task to a new worker
        self.K = K  # number of workers per task

    def assign(self, worker, golden_tasks, normal_tasks):
        """ Assign a task to a requesting worker based on the worker's trust score and reliability
        score """
        if worker.get_s() < self.tau and worker.get_r() < self.delta:
            g = self.alpha * (1 - worker.get_r()) + (1 - self.alpha) * worker.get_s()
            rand = Random()
            if rand.random() <= g:
                avail_tasks = set(golden_tasks)
                avail_tasks.difference_update(worker.get_labeled_pairs().keys())
                for task in avail_tasks:
                    if (worker.get_attacker_id() != -1) or (task in worker.get_pairs().keys()):
                        count = task.get_expose()
                        for curr_worker in task.get_assigned():
                            if curr_worker.get_r() < self.delta:
                                count += 1
                        if count < self.K:
                            task.assign(worker)
                            return task

        # assign a normal task
        for task in normal_tasks:
            assigned_workers = task.get_assigned()
            if worker not in assigned_workers:
                if worker.get_attacker_id() != -1:
                    task.assign(worker)
                    return task
                else:
                    if worker in task.get_workers():
                        task.assign(worker)
                        return task

        return None
