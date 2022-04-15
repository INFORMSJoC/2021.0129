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


def RSUCFLP(I, J, K, M_k, A_k, b_k, d_bar_k, d_uk, d_lk, Sigma_k, sigma_k, c, r, f, p_k):
    '''
    Robust Stochastic Uncapacitated Facility Location
    '''
    # Model
    model = gp.Model("uncap")
    
    # Variables
    y = model.addVars(I, vtype=GRB.BINARY, name="y")

    alpha_k = model.addVars(K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                            vtype=GRB.CONTINUOUS, name="alpha_k")

    beta_uk = model.addVars(J, K, lb=0, ub=+GRB.INFINITY,
                            vtype=GRB.CONTINUOUS, name="beta_uk")

    beta_lk = model.addVars(J, K, lb=0, ub=+GRB.INFINITY,
                            vtype=GRB.CONTINUOUS, name="beta_lk")

    rho_k = model.addVars(K, lb=0, ub=+GRB.INFINITY,
                          vtype=GRB.CONTINUOUS, name="rho_k")

    eta_k = model.addVars(K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                          vtype=GRB.CONTINUOUS, name="eta_k")

    xi_1 = model.addVars(M_k, K, lb=0, ub=+GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="xi_1")

    xi_2 = model.addVars(M_k, J, K, lb=0, ub=+GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="xi_2")

    xi_3 = model.addVars(M_k, I, J, K, lb=0, ub=+GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="xi_3")

    xi_4 = model.addVars(M_k, I, J, K, lb=0, ub=+GRB.INFINITY,
                         vtype=GRB.CONTINUOUS, name="xi_4")

    zeta_1 = model.addVars(J, K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                           vtype=GRB.CONTINUOUS, name="zeta_1")

    zeta_2 = model.addVars(J, J, K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                           vtype=GRB.CONTINUOUS, name="zeta_2")

    zeta_3 = model.addVars(J, I, J, K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                           vtype=GRB.CONTINUOUS, name="zeta_3")

    zeta_4 = model.addVars(J, I, J, K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                           vtype=GRB.CONTINUOUS, name="zeta_4")

    x_0 = model.addVars(I, J, K, lb=0, ub=0,
                        vtype=GRB.CONTINUOUS, name="x_0")

    x_1 = model.addVars(I, J, J, K, lb=-GRB.INFINITY, ub=+GRB.INFINITY,
                        vtype=GRB.CONTINUOUS, name="x_1")

    x_2 = model.addVars(I, J, K, lb=0, ub=0,
                        vtype=GRB.CONTINUOUS, name="x_2")
    
    # Constraints
    for k in range(K):
        term_1 = gp.LinExpr(b_k[:, k], xi_1.select("*", k))

        term_2_coef = np.dot(Sigma_k[:, :, k], d_bar_k[:, k])
        term_2 = gp.LinExpr(term_2_coef, zeta_1.select("*", k))

        term_3 = gp.LinExpr()
        for i in range(I):
            term_3.add(gp.LinExpr(c[i, :] - r[:], x_0.select(i, "*", k)))

        c1 = term_1
        c1.add(term_2, -1)
        c1.add(term_3)
        c1.addTerms(-1, alpha_k[k])

        model.addConstr(c1 <= 0, "c1_{}".format(k))

    for k in range(K):
        for ell in range(J):
            term_1 = gp.LinExpr(A_k[:, ell, k], xi_1.select("*", k))

            term_2 = gp.LinExpr(Sigma_k[:, ell, k], zeta_1.select("*", k))

            term_3 = gp.LinExpr()
            for i in range(I):
                term_3.add(gp.LinExpr(c[i, :] - r[:], x_1.select(i, "*", ell, k)))

            c2 = term_1
            c2.add(term_2, -1)
            c2.add(term_3, -1)
            c2.addTerms(-1, beta_lk[ell, k])
            c2.addTerms(1, beta_uk[ell, k])
            c2.addConstant(-1 * r[ell])

            model.addConstr(c2 >= 0, "c2_{}_{}".format(ell, k))

    for k in range(K):
        term_2_coef = np.dot(Sigma_k[:, :, k], d_bar_k[:, k])
        for j in range(J):
            term_1 = gp.LinExpr(b_k[:, k], xi_2.select("*", j, k))

            term_2 = gp.LinExpr(term_2_coef, zeta_2.select("*", j, k))

            c3 = term_1
            c3.add(term_2, -1)
            c3.add(x_0.sum("*", j, k))

            model.addConstr(c3 <= 0, "c3_{}_{}".format(j, k))

    for k in range(K):
        term_2_coef = np.dot(Sigma_k[:, :, k], d_bar_k[:, k])
        for j in range(J):
            for i in range(I):
                c4_term_1 = gp.LinExpr(b_k[:, k], xi_3.select("*", i, j, k))
                c5_term_1 = gp.LinExpr(b_k[:, k], xi_4.select("*", i, j, k))

                c4_term_2 = gp.LinExpr(term_2_coef, zeta_3.select("*", i, j, k))
                c5_term_2 = gp.LinExpr(term_2_coef, zeta_4.select("*", i, j, k))

                c4 = c4_term_1
                c4.add(c4_term_2, -1)
                c4.addTerms(1, x_0[i, j, k])

                c5 = c5_term_1
                c5.add(c5_term_2, -1)
                c5.addTerms(-1, x_0[i, j, k])

                model.addConstr(c4 <= 0, "c4_{}_{}_{}".format(i, j, k))
                model.addConstr(c5 <= 0, "c5_{}_{}_{}".format(i, j, k))

    for k in range(K):
        for j in range(J):
            for ell in range(J):
                term_1 = gp.LinExpr(A_k[:, ell, k], xi_2.select("*", j, k))

                term_2 = gp.LinExpr(Sigma_k[:, ell, k], zeta_2.select("*", ell, k))

                c6 = term_1
                c6.add(term_2, -1)
                c6.add(x_1.sum("*", j, ell, k), -1)

                if ell == j:
                    c6.addConstant(1)

                model.addConstr(c6 >= 0, "c6_{}_{}_{}".format(j, ell, k))

    for k in range(K):
        for i in range(I):
            for j in range(J):
                for ell in range(J):
                    term_1 = gp.LinExpr(A_k[:, ell, k], xi_3.select("*", i, j, k))

                    term_2 = gp.LinExpr(Sigma_k[:, ell, k], zeta_3.select("*", i, j, k))

                    c8 = term_1
                    c8.add(term_2, -1)
                    c8.addTerms(-1, x_1[i, j, ell, k])
                    if ell == j:
                        c8.addTerms(1, y[i])

                    model.addConstr(c8 >= 0, "c8_{}_{}_{}_{}".format(j, ell, i, k))

    for k in range(K):
        for i in range(I):
            for j in range(J):
                for ell in range(J):
                    term_1 = gp.LinExpr(A_k[:, ell, k], xi_4.select("*", i, j, k))

                    term_2 = gp.LinExpr(Sigma_k[:, ell, k], zeta_4.select("*", i, j, k))

                    c10 = term_1
                    c10.add(term_2, -1)
                    c10.addTerms(1, x_1[i, j, ell, k])

                    model.addConstr(c10 >= 0, "c10_{}_{}_{}_{}".format(j, ell, i, k))

    for k in range(K):
        term_3 = gp.LinExpr()
        for i in range(I):
            term_3.add(gp.LinExpr(c[i, :] - r[:], x_2.select(i, "*", k)))

        c11 = term_3
        c11.addTerms(1, eta_k[k])
        c11.addTerms(-1, rho_k[k])

        model.addConstr(c11 == 0, "c11_{}".format(k))

    for k in range(K):
        c12 = gp.QuadExpr()
        for j in range(J):
            c12.add(zeta_1[j, k] * zeta_1[j, k])

        c12.add(eta_k[k] * eta_k[k], -1)

        model.addConstr(c12 <= 0, "c12_{}".format(k))

    for k in range(K):
        for j in range(J):
            c13 = gp.QuadExpr()
            for j1 in range(J):
                c13.add(zeta_2[j1, j, k] * zeta_2[j1, j, k])

            c13.add(x_2.sum("*", j, k))

            model.addConstr(c13 <= 0, "c13_{}_{}".format(j, k))

    for k in range(K):
        for i in range(I):
            for j in range(J):
                c14 = gp.QuadExpr()
                for j1 in range(J):
                    c14.add(zeta_3[j1, i, j, k] * zeta_3[j1, i, j, k])

                c14.addTerms(1, x_2[i, j, k])

                model.addConstr(c14 <= 0, "c14_{}_{}_{}".format(i, j, k))

    for k in range(K):
        for i in range(I):
            for j in range(J):
                c15 = gp.QuadExpr()
                for j1 in range(J):
                    c15.add(zeta_4[j1, i, j, k] * zeta_4[j1, i, j, k])

                c15.addTerms(-1, x_2[i, j, k])

                model.addConstr(c15 <= 0, "c15_{}_{}_{}".format(i, j, k))

    term_1 = gp.LinExpr(f, y.select("*"))

    term_2 = gp.LinExpr()
    for k in range(K):
        term_tmp = gp.LinExpr(d_uk[:, k], beta_uk.select("*", k))
        term_tmp.add(gp.LinExpr(d_lk[:, k], beta_lk.select("*", k)), -1)
        term_tmp.addTerms(1, alpha_k[k])
        term_tmp.addTerms(sigma_k[k], rho_k[k])

        term_2.add(term_tmp, p_k[k])

    obj = term_1
    obj.add(term_2)

    model.setObjective(obj, sense=GRB.MINIMIZE)
    
    # optimization
    model.setParam("OutputFlag", 0)
    model.optimize()
    
    run_time = model.getAttr('Runtime')

    solx = model.getAttr('x', y)
    solx_list = []
    solx_a = np.zeros((I, 1))
    for i in range(I):
        if solx[i] > 0.9:
            solx_a[i, 0] = 1
            key_temp = '%d' % (i+1)
            solx_list.append(key_temp)
    
    objValue = model.objVal
    solx_x0 = model.getAttr('x', x_0)
    x0_a = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                x0_a[i, j, k] = solx_x0[i, j, k]

    solx_x1 = model.getAttr('x', x_1)
    x1_a = np.zeros((I, J, J, K))
    for i in range(I):
        for j in range(J):
            for j1 in range(J):
                for k in range(K):
                    x1_a[i, j, j1, k] = solx_x1[i, j, j1, k]

    solx_x2 = model.getAttr('x', x_2)
    x2_a = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                x2_a[i, j, k] = solx_x2[i, j, k]
    
    return solx_list, objValue, run_time


