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
from Func_DRO_SDRO_SAA import *

warnings.filterwarnings('ignore')

global file_nameg  # file_path
file_nameg = os.getcwd() + '\\'
I = 10
J = 20
N_k = 1
M_k = 2 * J

########################################### Parameters #####################################################

Parameters_dict = read_xlsx(file_nameg + 'Parameters.xlsx')
c = Parameters_dict['transportation_cost']
r = Parameters_dict['penalty'].reshape(J)
q = Parameters_dict['capacity'].reshape(I)
f = Parameters_dict['fixed_cost'].reshape(I)

############################################ S-DRO Model ##################################################

Parameters_SDRO_dict = read_xlsx(file_nameg + 'Parameters_SDRO.xlsx')
K_SDRO = 4
A_SDRO = np.zeros((M_k, J, K_SDRO))
A_SDRO[:, :, 0] = Parameters_SDRO_dict['Matrix_A_1']
A_SDRO[:, :, 1] = Parameters_SDRO_dict['Matrix_A_2']
A_SDRO[:, :, 2] = Parameters_SDRO_dict['Matrix_A_3']
A_SDRO[:, :, 3] = Parameters_SDRO_dict['Matrix_A_4']

b_SDRO = Parameters_SDRO_dict['Vector_b']

d_bar_SDRO = Parameters_SDRO_dict['d_bar']

d_uk_SDRO = Parameters_SDRO_dict['d_plus']

d_lk_SDRO = Parameters_SDRO_dict['d_minus']

Covariance_SDRO = np.zeros((J, J, K_SDRO))
Covariance_SDRO[:, :, 0] = Parameters_SDRO_dict['Covariance_1']
Covariance_SDRO[:, :, 1] = Parameters_SDRO_dict['Covariance_2']
Covariance_SDRO[:, :, 2] = Parameters_SDRO_dict['Covariance_3']
Covariance_SDRO[:, :, 3] = Parameters_SDRO_dict['Covariance_4']

sigma_SDRO = Parameters_SDRO_dict['sigma'].reshape(K_SDRO)

p_SDRO = Parameters_SDRO_dict['p'].reshape(K_SDRO)

obj_val_SDRO, y_star_SDRO, zeta_SDRO, eta_SDRO\
= CLP_EW_EXACT(I, J, K_SDRO, M_k, A_SDRO, b_SDRO, Covariance_SDRO, d_bar_SDRO, d_uk_SDRO, d_lk_SDRO, sigma_SDRO,
               c, f, p_SDRO, q, r)

index_zeta_1_SDRO = np.where(zeta_SDRO[:, 0] > 0.0001)
zeta_len_1_SDRO = len(index_zeta_1_SDRO[0])
zeta_1_SDRO = zeta_SDRO[index_zeta_1_SDRO, 0]
eta_1_SDRO = eta_SDRO[:, index_zeta_1_SDRO[0], 0]
demand_1_wc_SDRO = np.zeros((J, zeta_len_1_SDRO))
for index in range(zeta_len_1_SDRO):
    demand_1_wc_SDRO[:, index] = eta_1_SDRO[:, index]/zeta_1_SDRO[0, index]

index_zeta_2_SDRO = np.where(zeta_SDRO[:, 1] > 0.0001)
zeta_len_2_SDRO = len(index_zeta_2_SDRO[0])
zeta_2_SDRO = zeta_SDRO[index_zeta_2_SDRO, 1]
eta_2_SDRO = eta_SDRO[:, index_zeta_2_SDRO[0], 1]
demand_2_wc_SDRO= np.zeros((J, zeta_len_2_SDRO))
for index in range(zeta_len_2_SDRO):
    demand_2_wc_SDRO[:, index] = eta_2_SDRO[:, index]/zeta_2_SDRO[0, index]

index_zeta_3_SDRO = np.where(zeta_SDRO[:, 2] > 0.0001)
zeta_len_3_SDRO = len(index_zeta_3_SDRO[0])
zeta_3_SDRO = zeta_SDRO[index_zeta_3_SDRO, 2]
eta_3_SDRO = eta_SDRO[:, index_zeta_3_SDRO[0], 2]
demand_3_wc_SDRO = np.zeros((J, zeta_len_3_SDRO))
for index in range(zeta_len_3_SDRO):
    demand_3_wc_SDRO[:, index] = eta_3_SDRO[:, index]/zeta_3_SDRO[0, index]

index_zeta_4_SDRO = np.where(zeta_SDRO[:, 3] > 0.0001)
zeta_len_4_SDRO = len(index_zeta_4_SDRO[0])
zeta_4_SDRO = zeta_SDRO[index_zeta_4_SDRO, 3]
eta_4_SDRO = eta_SDRO[:, index_zeta_4_SDRO[0], 3]
demand_4_wc_SDRO = np.zeros((J, zeta_len_4_SDRO))
for index in range(zeta_len_4_SDRO):
    demand_4_wc_SDRO[:, index] = eta_4_SDRO[:, index]/zeta_4_SDRO[0, index]

demand_wc_SDRO = demand_1_wc_SDRO
demand_wc_SDRO = np.column_stack((demand_wc_SDRO, demand_2_wc_SDRO))
demand_wc_SDRO = np.column_stack((demand_wc_SDRO, demand_3_wc_SDRO))
demand_wc_SDRO = np.column_stack((demand_wc_SDRO, demand_4_wc_SDRO))

p_wc_SDRO = zeta_1_SDRO/K_SDRO
p_wc_SDRO = np.column_stack((p_wc_SDRO, zeta_2_SDRO/K_SDRO))
p_wc_SDRO = np.column_stack((p_wc_SDRO, zeta_3_SDRO/K_SDRO))
p_wc_SDRO = np.column_stack((p_wc_SDRO, zeta_4_SDRO/K_SDRO))

num_SDRO = p_wc_SDRO.shape[1]

OPT_SDRO_wc = np.zeros(num_SDRO)
TS_SDRO_wc = np.zeros(num_SDRO)
ST_SDRO_wc = np.zeros(num_SDRO)

y_SDRO = y_star_SDRO
for num in range(num_SDRO):
    obj_SDRO_wc, TSC_SDRO_wc, STC_SDRO_wc, solx_a_SDRO_wc = CFLP(y_SDRO, demand_wc_SDRO[:, num], f, I, J, c, r, q)
    OPT_SDRO_wc[num] = obj_SDRO_wc
    TS_SDRO_wc[num] = TSC_SDRO_wc
    ST_SDRO_wc[num] = STC_SDRO_wc
SDRO_TScost = np.dot(TS_SDRO_wc, p_wc_SDRO[0, :])
SDRO_STcost = np.dot(ST_SDRO_wc, p_wc_SDRO[0, :])

############################################### DRO Model ###################################################

Parameters_DRO_dict = read_xlsx(file_nameg + 'Parameters_DRO.xlsx')
K_DRO = 1

A_DRO = np.zeros((M_k, J, K_DRO))
A_DRO[:, :, 0] = Parameters_DRO_dict['Matrix_A']


b_DRO = Parameters_DRO_dict['Vector_b']

d_bar_DRO = Parameters_DRO_dict['d_bar']

d_uk_DRO = Parameters_DRO_dict['d_plus']

d_lk_DRO = Parameters_DRO_dict['d_minus']

Covariance_DRO = np.zeros((J, J, K_DRO))
Covariance_DRO[:, :, 0] = Parameters_DRO_dict['Covariance']

sigma_DRO = Parameters_DRO_dict['sigma'][0]

p_DRO = Parameters_DRO_dict['p'][0]


obj_val_DRO, y_star_DRO, zeta_DRO, eta_DRO \
    = CLP_EW_EXACT(I, J, K_DRO, M_k, A_DRO, b_DRO, Covariance_DRO, d_bar_DRO, d_uk_DRO, d_lk_DRO, sigma_DRO,
               c, f, p_DRO, q, r)

index_zeta_DRO = np.where(zeta_DRO[:, 0] > 0.0001)
zeta_len_DRO = len(index_zeta_DRO[0])
zeta_DRO_wc = zeta_DRO[index_zeta_DRO, 0]
eta_DRO_wc = eta_DRO[:, index_zeta_DRO[0], 0]
demand_DRO_wc = np.zeros((J, zeta_len_DRO))
for index in range(zeta_len_DRO):
    demand_DRO_wc[:, index] = eta_DRO_wc[:, index]/zeta_DRO_wc[0, index]

p_wc_DRO = zeta_DRO_wc/1

num_DRO = p_wc_DRO.shape[1]

OPT_DRO_wc = np.zeros(num_DRO)
TS_DRO_wc = np.zeros(num_DRO)
ST_DRO_wc = np.zeros(num_DRO)

y_DRO = y_star_DRO
for num in range(num_DRO):
    obj_DRO_wc, TSC_DRO_wc, STC_DRO_wc, solx_a_DRO_wc = CFLP(y_DRO, demand_DRO_wc[:, num], f, I, J, c, r, q)
    OPT_DRO_wc[num] = obj_DRO_wc
    TS_DRO_wc[num] = TSC_DRO_wc
    ST_DRO_wc[num] = STC_DRO_wc
DRO_TScost = np.dot(TS_DRO_wc, p_wc_DRO[0, :])
DRO_STcost = np.dot(ST_DRO_wc, p_wc_DRO[0, :])



########################################### SAA Model ######################################################

Parameters_SAA_dict = read_xlsx(file_nameg + 'Parameters_SAA.xlsx')
demand_SAA = Parameters_SAA_dict['demand'].T
N = 36
obj_val_SAA, y_SAA, SAA_TScost, SAA_STcost = SAA(I, J, N, demand_SAA, q, c, r, f)

########################################## Output Result ###################################################
sheet_name = "Result"
workbook = xlsxwriter.Workbook(file_nameg + 'Result_DRO_SDRO_SAA.xlsx')
worksheet = workbook.add_worksheet(sheet_name)
worksheet.write(0, 0, 'Model')
worksheet.write(1, 0, 'DRO')
worksheet.write(2, 0, 'S-DRO')
worksheet.write(3, 0, 'SAA')
y_DRO_list = []
for i in range(I):
    if y_star_DRO[i] == 1:
        y_DRO_list.append(str(i+1))
y_SDRO_list = []
for i in range(I):
    if y_star_SDRO[i] == 1:
        y_SDRO_list.append(str(i+1))
y_SAA_list = []
for i in range(I):
    if y_SAA[i] == 1:
        y_SAA_list.append(str(i+1))
worksheet.write(0, 1, 'Optimal location design')
worksheet.write(1, 1, '_'.join(y_DRO_list))
worksheet.write(2, 1, '_'.join(y_SDRO_list))
worksheet.write(3, 1, '_'.join(y_SAA_list))
worksheet.write(0, 2, 'Total capacity')
worksheet.write(1, 2, np.dot(q, y_star_DRO))
worksheet.write(2, 2, np.dot(q, y_star_SDRO))
worksheet.write(3, 2, np.dot(q, y_SAA))
worksheet.write(0, 3, 'Expected total cost')
worksheet.write(1, 3, obj_val_DRO)
worksheet.write(2, 3, obj_val_SDRO)
worksheet.write(3, 3, obj_val_SAA)
worksheet.write(0, 4, 'Expected transportation cost')
worksheet.write(1, 4, DRO_TScost)
worksheet.write(2, 4, SDRO_TScost)
worksheet.write(3, 4, SAA_TScost)
worksheet.write(0, 5, 'Expected shortage cost')
worksheet.write(1, 5, DRO_STcost)
worksheet.write(2, 5, SDRO_STcost)
worksheet.write(3, 5, SAA_STcost)
worksheet.write(0, 6, 'Fixed cost')
worksheet.write(1, 6, np.dot(f, y_star_DRO))
worksheet.write(2, 6, np.dot(f, y_star_SDRO))
worksheet.write(3, 6, np.dot(f, y_SAA))
workbook.close()