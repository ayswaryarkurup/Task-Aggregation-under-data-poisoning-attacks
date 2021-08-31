# Provides the modeling of workers. Each worker is associated with a trust score, a reliability score and an accuracy on golden tasks.
# The attacker id indicates which data poisoning attacker controls the worker.


class Worker:

    def __init__(self):
        self.pairs = {}  # the (task, label) pairs in the original dataset
        self.labeled_pairs = {}  # the (task, label) pairs for tasks currently assigned to the worker
        self.s_j = 0  # Trust score
        self.r_j = 0  # reliability score
        self.p_j = 0  # accuracy on golden tasks
        self.attacker_id = -1  # indicates which attacker the worker belongs to 
        self.weight = 0  # weight of the worker's labels in truth inference
        self.banned = False  # indicates whether the worker is banned

    def add_pair(self, task, label):
        """ add a (task, label) pair in the original dataset """
        self.pairs[task] = label

    def get_pairs(self):
        """ return the (task, label) pairs in the original dataset """
        return self.pairs

    def label(self, task, label):
        """ update the current (task, label) pair """
        self.labeled_pairs[task] = label

    def get_labeled_pairs(self):
        """ return the current (task, label) pairs """
        return self.labeled_pairs

    def remove(self, task):
       #remove the label of a task labeled by the banned worker 
        del self.labeled_pairs[task]

    def set_s(self, s):
        # set the trust score
        self.s_j = s

    def get_s(self):
        # return the trust score 
        return self.s_j

    def set_r(self, r):
        """ set the reliability score """
        self.r_j = r

    def get_r(self):
        """ return the reliability score """
        return self.r_j

    def set_p(self, p):
        """ set the accuracy on golden tasks """
        self.p_j = p

    def get_p(self):
        """ return the accuracy on golden tasks """
        return self.p_j

    def set_attacker_id(self, attacker_id):
       #set the attacker ID (id=-1 means the worker is an independent worker) 
        self.attacker_id = attacker_id

    def get_attacker_id(self):
        """ return the attacker ID """
        return self.attacker_id

    def set_weight(self, w):
        """ update the worker's weight in  truth inference """
        self.weight = w

    def get_weight(self):
        """ return the worker's weight in truth inference """
        return self.weight

    def ban(self):
        """ ban the worker """
        self.banned = True

    def is_banned(self):
        """ check whether the worker is banned """
        return self.banned

    def reset(self):
        """ reset the features of the worker for a new run """
        self.labeled_pairs = {}
        self.s_j = 0
        self.r_j = 0
        self.attacker_id = -1
        self.weight = 0.8
        self.banned = False
