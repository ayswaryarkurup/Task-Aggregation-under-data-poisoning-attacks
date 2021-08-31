import math
import os
import sys
import traceback

from preprocess import Preprocess
from tdadp import Tdadp


def _output_to_console_and_file(lines, file):
    for line in lines:
        print(line)
    for line in lines:
        file.write(line + '\n')


# noinspection PyPep8Naming
def main(args):
    if len(args) != 9 and len(args) != 14:
        print("Invalid number of parameters")
        sys.exit(0)

    dataset = args[0]
    run_num = int(args[1])
    B = int(args[2])
    alpha = float(args[3])
    tau = float(args[4])
    delta = float(args[5])
    mu = float(args[6])
    epsilon = float(args[7])
    # noinspection SpellCheckingInspection
    lambd = int(args[8])

    if len(args) == 9:
        pre = Preprocess.real_dataset(dataset, run_num, mu, epsilon, lambd)
        pre.read_data()
        pre.formalize()
    else:
        N = int(args[9])
        M = int(args[10])
        L = int(args[11])
        K = int(args[12])
        theta = float(args[13])

        pre = Preprocess.synth_dataset(dataset, run_num, mu, epsilon, lambd, N, M, L, K, theta)
        pre.simulate()
        pre.formalize()

    try:
        file_path = os.sep.join([dataset, 'result.txt'])
        with open(file_path, 'w') as file:
            # write worker number M, task number N, label size L and worker number per task K
            file.write(f'{dataset}\n')
            # noinspection SpellCheckingInspection
            tdssa = Tdadp(B, alpha, tau, delta)
            tdssa.read_normal(dataset)

            accuracy = [0] * run_num  # record aggregation accuracy in each run
            exposed = [0] * run_num  # record number of exposed golden tasks in each run
            cost = [0] * run_num  # record number of golden tasks for testing each worker in each run
            _time = [0] * run_num  # record running time in each run

            ave_a_accuracy = 0  # average aggregation accuracy
            ave_e_number = 0  # average number of exposed golden tasks
            ave_t_cost = 0  # average number of golden tasks for testing each worker
            ave_running_time = 0  # average running time

            for r in range(0, run_num):
                tdssa.read_golden(dataset)
                tdssa.read_attack(dataset, r)
                tdssa.read_order(dataset, r)
                tdssa.run()
                accuracy[r] = tdssa.get_a_accuracy()
                exposed[r] = tdssa.get_e_number()
                cost[r] = tdssa.get_t_cost()
                _time[r] = tdssa.get_running_time()
                ave_a_accuracy += tdssa.get_a_accuracy()
                ave_e_number += tdssa.get_e_number()
                ave_t_cost += tdssa.get_t_cost()
                ave_running_time += tdssa.get_running_time()
                out = f'Run {(r + 1)} --- A-Accuracy:{accuracy[r]}   T-Cost:{cost[r]}  Time:{_time[r]}ms'
                _output_to_console_and_file([out], file)

            ave_a_accuracy /= run_num
            ave_e_number /= run_num
            ave_t_cost /= run_num
            ave_running_time /= run_num

            std_a_accuracy = 0  # standard error of A-Accuracy
            std_e_number = 0  # standard error of E-Number
            std_t_cost = 0  # standard error of T-Cost
            std_running_time = 0  # standard error of Time

            for r in range(0, run_num):
                std_a_accuracy += math.pow(ave_a_accuracy - accuracy[r], 2)
                std_e_number += math.pow(ave_e_number - exposed[r], 2)
                std_t_cost += math.pow(ave_t_cost - cost[r], 2)
                std_running_time += math.pow(ave_running_time - _time[r], 2)

            std_a_accuracy = math.sqrt(std_a_accuracy / (run_num - 1)) / math.sqrt(run_num)
            std_e_number = math.sqrt(std_e_number / (run_num - 1)) / math.sqrt(run_num)
            std_t_cost = math.sqrt(std_t_cost / (run_num - 1)) / math.sqrt(run_num)
               # (long) (Math.sqrt(std_running_time/(run_num - 1))/Math.sqrt(run_num))
            std_running_time = int(math.sqrt(std_running_time // (run_num - 1)) / math.sqrt(run_num))
            output_lines = [
                '\nAverage: ',
                f'A-Accuracy: {ave_a_accuracy}  Standard Error: {std_a_accuracy}',
              #  f'E-Number:{ave_e_number}  Standard Error: {std_e_number}',
                f'T-cost:{ave_t_cost}  Standard Error: {std_t_cost}',
                f'Time:{ave_running_time}ms  Standard Error: {std_running_time}'
            ]
            _output_to_console_and_file(output_lines, file)
    except Exception as e:
        print(e)
        traceback.print_stack()
        traceback.print_exc()


if __name__ == '__main__':
    main(sys.argv[1:])
