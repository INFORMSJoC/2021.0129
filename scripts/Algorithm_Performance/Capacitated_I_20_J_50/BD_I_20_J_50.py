from Func_BD_I_20_J_50 import *

# file_path
file_nameg = os.getcwd() + '\\'

I_all = [20]
J_all = [50]
K_all = [1, 3, 5]

for i in range(len(I_all)):
    for j in range(len(J_all)):
        for k in range(len(K_all)):
            print('*'*60)

            I = I_all[i]
            J = J_all[j]
            K = K_all[k]
            print('='*60)
            print('           BD: Capacitated_Instance_I_{}_J_{}_K_{}'.format(I, J, K))
            print('='*60)

            instance_name = 'Capacitated_I_' + str(I) + '_J_' + str(J) + '_K_' + str(K) + '.npy'

            instance_dict = np.load(instance_name, allow_pickle=True).item()
            M_k = 2 * J
            f = instance_dict['fixed_cost']
            q = instance_dict['capacity']
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

            obj_val_all, y_star_all, alpha_star_all, Omega_all, \
            UB, LB, UB_all, \
            run_time_dec_RMP_all, run_time_MICP_all, time_loop_all, \
            cycle_all, gap_all, gap_prime_all, run_time_all, time_all, run_time_QLP \
                = BD_I20_J50(I, J, K, M_k, A, b, COV_inverse_sqrt, d_bar,
                             d_plus, d_minus, sigma, c, f, p, q, r, file_nameg)



