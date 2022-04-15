import sys
import os
import math
import datetime
import pickle
import time
import xlsxwriter
from math import *
from numpy import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from openpyxl import load_workbook
import warnings
from Func_CPU_Time_Uncapacitated_Problem import *

# file_path
file_nameg = os.getcwd() + '\\'

I_all = [10, 20]
J_all = [10, 30, 50]
K_all = [1, 3, 5]

CPU_time_dict = {}
expected_total_cost_dict = {}
opt_solution_dict = {}

for i in range(len(I_all)):
    for j in range(len(J_all)):
        for k in range(len(K_all)):

            I = I_all[i]
            J = J_all[j]
            K = K_all[k]
            M_k = 2 * J

            print('*'*60)
            print('='*60)
            print('        Uncapacitated problem instance I_{}_J_{}_K_{}'.format(I, J, K))
            print('='*60)

            instance_name = 'Uncapacitated_I_' + str(I) + '_J_' + str(J) + '_K_' + str(K) + '.npy'

            instance_dict = np.load(instance_name, allow_pickle=True).item()

            f = instance_dict['fixed_cost']
            c = instance_dict['transportation_cost']
            r = instance_dict['penalty']
            A = instance_dict['A']
            b = instance_dict['b']
            d_bar = instance_dict['d_bar']
            d_plus = instance_dict['d_plus']
            d_minus = instance_dict['d_minus']
            COV_inverse_sqrt = instance_dict['COV_inverse_sqrt']
            sigma = instance_dict['sigma']
            p = instance_dict['p']

            solution, objValue, run_time = \
                RSUCFLP(I, J, K, M_k, A, b, d_bar, d_plus, d_minus, COV_inverse_sqrt, sigma, c, r, f, p)

            print('Computation time: {}'.format(run_time))
            print('Expected total cost: {}'.format(objValue))
            print('Optimal solution: {}'.format(solution))

            key = 'key_'+str(I)+'_'+str(J)+'_'+str(K)

            CPU_time_dict[key] = run_time
            expected_total_cost_dict[key] = objValue
            opt_solution_dict[key] = solution

workbook = xlsxwriter.Workbook(file_nameg + 'CPU_time_uncapacitated_problem.xlsx')
worksheet = workbook.add_worksheet('result')
worksheet.write(0, 0, '(|I|,|J|,|K|)')
worksheet.write(0, 1, 'Expected total cost')
worksheet.write(0, 2, 'Time')
count = 0
for i in range(len(I_all)):
    for j in range(len(J_all)):
        for k in range(len(K_all)):
            count += 1
            I = I_all[i]
            J = J_all[j]
            K = K_all[k]
            key = 'key_'+str(I)+'_'+str(J)+'_'+str(K)
            worksheet.write(count, 0, str(I)+'_'+str(J)+'_'+str(K))
            worksheet.write(count, 1, expected_total_cost_dict[key])
            worksheet.write(count, 2, CPU_time_dict[key])
workbook.close()


