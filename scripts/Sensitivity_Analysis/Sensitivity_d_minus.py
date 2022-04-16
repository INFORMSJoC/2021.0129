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
from Func_Sensitivity import *

warnings.filterwarnings('ignore')

global file_nameg  # file_path
file_nameg = os.getcwd() + '\\'

if __name__ == "__main__":
    K = 4
    I = 10
    J = 20
    N = 9
    N_k = 1
    M_k = 2 * J

    Parameters_dict = read_xlsx(file_nameg + 'Parameters.xlsx')
    c = Parameters_dict['transportation_cost']
    r = Parameters_dict['penalty'].reshape(J)
    q = Parameters_dict['capacity'].reshape(I)
    f = Parameters_dict['fixed_cost'].reshape((I,1))

    Parameters_SDRO_dict = read_xlsx(file_nameg + 'Parameters_SDRO_Sensitivity.xlsx')

    A_k = np.zeros((M_k, J, K))
    A_k[:, :, 0] = Parameters_SDRO_dict['Matrix_A_1']
    A_k[:, :, 1] = Parameters_SDRO_dict['Matrix_A_2']
    A_k[:, :, 2] = Parameters_SDRO_dict['Matrix_A_3']
    A_k[:, :, 3] = Parameters_SDRO_dict['Matrix_A_4']

    b_k = Parameters_SDRO_dict['Vector_b']

    d_bar_k = Parameters_SDRO_dict['d_bar']

    d_uk = Parameters_SDRO_dict['d_plus']

    d_lk = Parameters_SDRO_dict['d_minus']

    Sigma_k = np.zeros((J, J, K))
    Sigma_k[:, :, 0] = Parameters_SDRO_dict['Covariance_1']
    Sigma_k[:, :, 1] = Parameters_SDRO_dict['Covariance_2']
    Sigma_k[:, :, 2] = Parameters_SDRO_dict['Covariance_3']
    Sigma_k[:, :, 3] = Parameters_SDRO_dict['Covariance_4']

    sigma_k = Parameters_SDRO_dict['sigma'].reshape(K)

    p = Parameters_SDRO_dict['p'].reshape(K)

    rho = Parameters_SDRO_dict['rho']

    beta_l = Parameters_SDRO_dict['beta_l']

    beta_u = Parameters_SDRO_dict['beta_u']

    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])

    obj = 1516336.718
    trans = obj - np.dot(f[:, 0], y)

    delta = [0.01, 0.02, 0.03, 0.04, 0.05]
    index_num = 5


    opt_d_low_up = np.zeros((index_num, K))
    opt_d_low_up_delta = np.zeros((index_num, K))
    d_low_up_bound = np.zeros((index_num, K))
    opt_d_low_low = np.zeros((index_num, K))
    opt_d_low_low_delta = np.zeros((index_num, K))
    d_low_low_bound = np.zeros((index_num, K))

    for index in range(index_num):
        ############################################################################# d_low_up ####################################################################
        d_low_delta_1 = np.array([1*(1+delta[index]), 1, 1, 1])
        d_low_1 = np.multiply(d_lk, d_low_delta_1)
        index_neg_1 = np.where(d_uk - d_low_1 < 0)
        if len(index_neg_1[0] != 0):
            row_1 = index_neg_1[0]
            col_1 = index_neg_1[1]
            for i_index in range(row_1.size):
                d_low_1[row_1[i_index], col_1[i_index]] = d_uk[row_1[i_index], col_1[i_index]].copy() - 0.02
        opt_d_low_up[index, 0] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_1, sigma_k, c, f, p, q, r)
        opt_d_low_up[index, 0] = opt_d_low_up[index, 0] - np.dot(f[:, 0], y)
        opt_d_low_up_delta[index, 0] = opt_d_low_up[index, 0] - trans
        for k in range(K):
            d_low_up_bound[index, 0] += p[k] * np.dot((np.multiply(d_lk, np.array([1*(1+delta[index]), 1, 1, 1]))[:,k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_2 = np.array([1, 1*(1+delta[index]), 1, 1])
        d_low_2 = np.multiply(d_lk, d_low_delta_2)
        index_neg_2 = np.where(d_uk - d_low_2 < 0)
        if len(index_neg_2[0] != 0):
            row_2 = index_neg_2[0]
            col_2 = index_neg_2[1]
            for i_index in range(row_2.size):
                d_low_2[row_2[i_index], col_2[i_index]] = d_uk[row_2[i_index], col_2[i_index]].copy() - 0.02
        opt_d_low_up[index, 1] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_2, sigma_k, c, f, p, q, r)
        opt_d_low_up[index, 1] = opt_d_low_up[index, 1] - np.dot(f[:, 0], y)
        opt_d_low_up_delta[index, 1] = opt_d_low_up[index, 1] - trans
        for k in range(K):
            d_low_up_bound[index, 1] += p[k] * np.dot((np.multiply(d_lk, np.array([1, 1*(1+delta[index]), 1, 1]))[:, k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_3 = np.array([1, 1, 1*(1+delta[index]), 1])
        d_low_3 = np.multiply(d_lk, d_low_delta_3)
        index_neg_3 = np.where(d_uk - d_low_3 < 0)
        if len(index_neg_3[0] != 0):
            row_3 = index_neg_3[0]
            col_3 = index_neg_3[1]
            for i_index in range(row_3.size):
                d_low_3[row_3[i_index], col_3[i_index]] = d_uk[row_3[i_index], col_3[i_index]].copy() - 0.02
        opt_d_low_up[index, 2] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_3, sigma_k, c, f, p, q, r)
        opt_d_low_up[index, 2] = opt_d_low_up[index, 2] - np.dot(f[:, 0], y)
        opt_d_low_up_delta[index, 2] = opt_d_low_up[index, 2] - trans
        for k in range(K):
            d_low_up_bound[index, 2] += p[k] * np.dot((np.multiply(d_lk, np.array([1, 1, 1*(1+delta[index]), 1]))[:, k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_4 = np.array([1, 1, 1, 1*(1+delta[index])])
        d_low_4 = np.multiply(d_lk, d_low_delta_4)
        index_neg_4 = np.where(d_uk - d_low_4 < 0)
        if len(index_neg_4[0] != 0):
            row_4 = index_neg_4[0]
            col_4 = index_neg_4[1]
            for i_index in range(row_4.size):
                d_low_4[row_4[i_index], col_4[i_index]] = d_uk[row_4[i_index], col_4[i_index]].copy() - 0.01
        opt_d_low_up[index, 3] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_4, sigma_k, c, f, p, q, r)
        opt_d_low_up[index, 3] = opt_d_low_up[index, 3] - np.dot(f[:, 0], y)
        opt_d_low_up_delta[index, 3] = opt_d_low_up[index, 3] - trans
        for k in range(K):
            d_low_up_bound[index, 3] += p[k] * np.dot((np.multiply(d_lk, np.array([1, 1, 1, 1*(1+delta[index])]))[:, k] - d_lk[:, k]), beta_l[:, k])

        ########################################################################### d_low_low ##############################################################
        d_low_delta_1 = np.array([1*(1-delta[index]), 1, 1, 1])
        d_low_1 = np.multiply(d_lk, d_low_delta_1)
        opt_d_low_low[index, 0] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_1, sigma_k, c, f, p, q, r)
        opt_d_low_low[index, 0] = opt_d_low_low[index, 0] - np.dot(f[:, 0], y)
        opt_d_low_low_delta[index, 0] = opt_d_low_low[index, 0] - trans
        for k in range(K):
            d_low_low_bound[index, 0] += p[k] * np.dot((d_low_1[:, k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_2 = np.array([1, 1*(1-delta[index]), 1, 1])
        d_low_2 = np.multiply(d_lk, d_low_delta_2)
        opt_d_low_low[index, 1] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_2, sigma_k, c, f, p, q, r)
        opt_d_low_low[index, 1] = opt_d_low_low[index, 1] - np.dot(f[:, 0], y)
        opt_d_low_low_delta[index, 1] = opt_d_low_low[index, 1] - trans
        for k in range(K):
            d_low_low_bound[index, 1] += p[k] * np.dot((d_low_2[:, k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_3 = np.array([1, 1, 1*(1-delta[index]), 1])
        d_low_3 = np.multiply(d_lk, d_low_delta_3)
        opt_d_low_low[index, 2] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_3, sigma_k, c, f, p, q, r)
        opt_d_low_low[index, 2] = opt_d_low_low[index, 2] - np.dot(f[:, 0], y)
        opt_d_low_low_delta[index, 2] = opt_d_low_low[index, 2] - trans
        for k in range(K):
            d_low_low_bound[index, 2] += p[k] * np.dot((d_low_3[:, k] - d_lk[:, k]), beta_l[:, k])

        d_low_delta_4 = np.array([1, 1, 1, 1*(1-delta[index])])
        d_low_4 = np.multiply(d_lk, d_low_delta_4)
        opt_d_low_low[index, 3] = CLP_EW_EXACT_fixy(y, I, J, K, M_k, A_k, b_k, Sigma_k, d_bar_k, d_uk, d_low_4, sigma_k, c, f, p, q, r)
        opt_d_low_low[index, 3] = opt_d_low_low[index, 3] - np.dot(f[:, 0], y)
        opt_d_low_low_delta[index, 3] = opt_d_low_low[index, 3] - trans
        for k in range(K):
            d_low_low_bound[index, 3] += p[k] * np.dot((d_low_4[:, k] - d_lk[:, k]), beta_l[:, k])

    delta_per = ['5%', '4%', '3%', '2%', '1%', '-1%', '-2%', '-3%', '-4%', '-5%']
    workbook = xlsxwriter.Workbook(file_nameg + 'Result_sensitivity_d_minus.xlsx')
    worksheet = workbook.add_worksheet('Result')
    worksheet.write(0, 0, 'delta')
    worksheet.write(0, 1, 'State k=1')
    worksheet.write(0, 3, 'State k=2')
    worksheet.write(0, 5, 'State k=3')
    worksheet.write(0, 7, 'State k=4')
    worksheet.write(1, 1, '\Delta(y|d^-_k)')
    worksheet.write(1, 2, 'UB')
    worksheet.write(1, 3, '\Delta(y|d^-_k)')
    worksheet.write(1, 4, 'UB')
    worksheet.write(1, 5, '\Delta(y|d^-_k)')
    worksheet.write(1, 6, 'UB')
    worksheet.write(1, 7, '\Delta(y|d^-_k)')
    worksheet.write(1, 8, 'UB')
    for num_delta in range(10):
        worksheet.write(2+num_delta, 0, delta_per[num_delta])
    for k in range(K):
        for i in range(5):
            worksheet.write((2+i), (1+2*k), opt_d_low_up_delta[(4-i), k])
    for k in range(K):
        for i in range(5):
            worksheet.write((2+i), (2+2*k), d_low_up_bound[(4-i), k])
    for k in range(K):
        for i in range(5):
            worksheet.write((7+i), (1+2*k), opt_d_low_low_delta[i, k])
    for k in range(K):
        for i in range(5):
            worksheet.write((7+i), (2+2*k), d_low_low_bound[i, k])
    workbook.close()
