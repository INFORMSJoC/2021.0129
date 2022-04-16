[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Robust Stochastic Facility Location: Sensitivity Analysis and Exact Solution

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper "Robust Stochastic Facility Location: Sensitivity Analysis and Exact Solution" by T. Liu, F. Saldanha-da-Gama, S. Wang and Y. Mao. 

## Cite

To cite this material, please cite this repository, using the following DOI.

[![DOI](https://zenodo.org/badge/285853815.svg)](https://zenodo.org/badge/latestdoi/285853815)

Below is the BibTex for citing this version of the code.

```
@article{RSFL,
  author =        {Tianqi Liu, Francisco Saldanha-da-Gama, Shuming Wang and Yuchen Mao},
  publisher =     {INFORMS Journal on Computing},
  title =         {Data for Robust Stochastic Facility Location: Sensitivity Analysis and Exact Solution},
  year =          {2022},
  doi =           {},
  url =           {https://github.com/INFORMSJoC/2021.0129},
}  
```

## Requirements
For these experiments, we use
* Python (the codes are written in Python 3.8)
* Gurobi 9.1.2

## Content
This repository includes the source code and computational results for all the experiments presented in the paper.

### Data files
The folder **data** contains all the parameters and samples used in our experiments.
1. In the folder **Solution_DRO_SDRO_SAA**, the file [Parameters.xslx](data/Solution_DRO_SDRO_SAA/Parameters.xlsx) includes the fixed cost, capacity, transportation cost ,and penalty cost, which are used in the experiments of Section 6.1. Furthermore, the files [Parameters_DRO.xlsx](data/Parameters_DRO.xlsx), [Parameters_SAA](data/Parameters_SAA.xlsx) and [Parameters_SDRO](data/Parameters_SDRO.xlsx) include the parameters of the DRO model, SAA model and S-DRO model proposed in our paper, respectively.  
2. The file [d_sample_set_seasonality.xlsx](data/Out_of_Sample_seasonality/d_sample_set_seasonality.xlsx) in the folder **Out_of_Sample_seasonality** contains 100 sets of randomly generated demand samples with seasonal information. 
3. The file [d_sample_set_nonseasonality.xlsx](data/Out_of_Sample_nonseasonality/d_sample_set_nonseasonality.xlsx) in the folder **Out_of_Sample_nonseasonality** includes 100 sets of randomly generated demand samples without seasonal information.
4. The file [demand_observation_2019.xlsx](data/Real_Case_2019/demand_observation_2019.xlsx) in the folder **Real_Case_2019** contains the true observation of the demand in each quarter of 2019.
5. In the folder **Out_of_Sample_Robustness**, the files [d_sample_set_phi_0.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_0.xlsx),...,[d_sample_set_phi_5.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_5.xlsx) contain 100 sets of randomly generated demand samples with different proportion of the worst-case distribution \phi \in {0.0, 0.1, 0.3, 0.5}, respectively.
6. In the folder **CPU_Time_Uncapacitated_Problem**, the files [Uncapacitated_I_10_J_10_K_1.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_10_J_10_K_1.npy),..., [Uncapacitated_I_20_J_50_K_5.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_20_J_50_K_5.npy) contain the parameters of the uncapacitated S-DRO model with the different number of the potential location sites I, customer sites J and states K. 
7. The files [Capacitated_I_10_J_10_K_1.npy](data/Algorithm_Performance/Capacitated_I_10_J_10/Capacitated_I_10_J_10_K_1.npy),...,[Capacitated_I_20_J_50_K_5.npy](data/Algorithm_Performance/Capacitated_I_20_J_50/Capacitated_I_20_J_50_K_5.npy) in the folder **Algorithm_Performance** contain the parameters of the capacitated S-DRO model with the different number of the potential location sites I, customer sites J and states K.

### Code files 

The folder **scripts** includes all the codes used in our experiments.
1. The code files in the folder **Solution_DRO_SDRO_SAA** are for obtaining the solution profiles for the DRO, S-DRO and SAA models, where the file [Solution_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Solution_DRO_SDRO_SAA.py) is the main program and [Func_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Func_DRO_SDRO_SAA.py) contains all the functions used in the main program. The codes have been used in Section 6.1 (Value of state-wise distributional information) and 6.2 (Value of robustness) in our paper.
2. The code file [Out_of_Sample_seasonality.py](scripts/Out_of_Sample_seasonality/Out_of_Sample_seasonality.py) in the folder **Out_of_Sample_seasonality** is for evaluating the performance of the S-DRO solution compared with the DRO solution using the out-of-sample tests with the seasonal information, which has been used in Section 6.1.
3. The code file [Real_Case_2019.py](scripts/Real_Case_2019/Real_Case_2019.py) in the folder **Real_Case_2019** is for evaluating the performance of the S-DRO and DRO solutions using the true demand observations in 2019, which has been used in Section 6.1.
4. The code file [Out_of_Sample_nonseasonality.py](scripts/Out_of_Sample_nonseasonality/Out_of_Sample_nonseasonality.py) in the folder **Out_of_Sample_nonseasonality** is for testing the performance of both S-DRO and DRO solutions under the out-of-sample scenarios without seasonality, which has been used in Section 6.1.
5. The code file [Out_of_Sample_Robustness.py](scripts/Out_of_Sample_Robustness/Out_of_Sample_Robustness.py) in the folder **Out_of_Sample_Robustness** is for  evaluating the value of robustness of the proposed S-DRO model and the DRO model by comparing their out-of-sample performance with the SAA counterpart, which has been used in Section 6.2.
6. The code files in the folder **Sensitivity_Analysis** is for evaluating the changes of worst-case expected transportation cost in different states with respect to the perturbation of the ambiguity parameters \sigma_k, d^+_k and d^-_k, where the files [Sensitivity_sigma.py](scripts/Sensitivity_Analysis/Sensitivity_sigma.py), [Sensitivity_d_plus.py](scripts/Sensitivity_Analysis/Sensitivity_d_plus.py) and [Sensitivity_d_minus.py](scripts/Sensitivity_Analysis/Sensitivity_d_minus.py) are the main programs for different parameters and the file [Func_Sensitivity.py](scripts/Sensitivity_Analysis/Func_Sensitivity.py) contains all the functions used in the main programs. The codes have been used in Section 6.3 (Sensitivity analysis of ambiguity set parameters) in our paper.
7. The code files in the folder **CPU_Time_Uncapacitated_Problem** are for evaluating the computational performance of solving the uncapacitated S-DRO model exactly in the reformulation of the mixed-interger second-order cone program, where the file [CPU_Time_Uncapacitated_Problem.py](scripts/CPU_Time_Uncapacitated_Problem/CPU_Time_Uncapacitated_Problem.py) is the main program and the file [Func_CPU_Time_Uncapacitated_Problem.py](scripts/CPU_Time_Uncapacitated_Problem/Func_CPU_Time_Uncapacitated_Problem.py) contains all the functions used in the main program. The codes in this folder have been used in Section 6.4 (Performance of the exact solution approach) in our paper.
8. The code files in the folder **Algorithm_Performance** is for comparing the subgradient-based Nested Benders decomposition algorithm (S-NBD) and the Benders decomposition approach (BD) under different instances. The files [S_NBD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/S_NBD_I_10_J_10.py),...,[S_NBD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/S_NBD_I_20_J_50.py) are the main programs of the S-NBD under different instances and the files [Func_S_NBD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/Func_S_NBD_I_10_J_10.py),...,[Func_S_NBD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/Func_S_NBD_I_20_J_50.py) includes all the functions used in the main programs. Similarly, the files [BD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/BD_I_10_J_10.py),...,[BD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/BD_I_20_J_50.py) are the main programs of BD under different instances and the files [Func_BD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/Func_BD_I_10_J_10.py),...,[Func_BD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/Func_BD_I_20_J_50.py) contains all the functions used in the main program. The codes have been used in Section 6.4. 

### Results files

The folder **results** contains the results for all numerical experiments used in our paper.
1. The file [Result_DRO_SDRO_SAA.xlsx](/results/Solution_DRO_SDRO_SAA/Result_DRO_SDRO_SAA.xlsx) in the folder **Solution_DRO_SDRO_SAA** includes the solution profiles for the DRO, S-DRO and SAA models, which are exactly Table 1 and Table 2 in our paper.
2. The file [out_of_sample_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_total_cost_seasonality.xlsx) in the folder **Out_of_Sample_seasonality** includes the out-of-sample total cost of both S-DRO and DRO location designs across 100 out-of-sample tests with seasonal information. Similarly, the files  [out_of_sample_conservativeness_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_total_cost_seasonality.xlsx) and [out_of_sample_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_transportation_cost_seasonality.xlsx) contain the out-of-sample conservativeness and out-of-sample transportation cost, respectively. Furthermore, the files [Ave_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_total_cost_seasonality.xlsx), [Ave_conservativeness_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_conservativeness_seasonality.xlsx) and [Ave_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_transportation_cost_seasonality.xlsx) include the averaged out-of-sample total cost, conservativeness value and transportation cost of each test, respectively. The above results have been recorded in Figure 3 and Figure 4. Finally, the file [Diff_Ave_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Diff_Ave_total_cost_seasonality.xlsx) includes the difference between the average total cost corresponding to
the DRO solution and that for the S-DRO solution, as well as the file [Diff_Ave_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Diff_Ave_transportation_cost_seasonality.xlsx) contains the differences between the average transportation costs for the DRO and the S-DRO solutions. The results have been presented in Figure 6.
3. The file [Result_real_case_2019.xlsx](results/Real_Case_2019/Result_real_case_2019.xlsx) in the folder **Real_Case_2019** records the total costs and the associate out-of-sample conservativeness values in each quarter of 2019 under both S-DRO and DRO solutions, which has been plotted in Figure 5.
4. The files [Diff_Ave_total_cost_nonseasonality.xlsx](results/Out_of_Sample_nonseasonality/Diff_Ave_total_cost_nonseasonality.xlsx) and [Diff_Ave_transportation_cost_nonseasonality.xlsx](results/Out_of_Sample_nonseasonality/Diff_Ave_transportation_cost_nonseasonality.xlsx) in the folder **Out_of_Sample_nonseasonality** include the averaged difference in out-of-sample total cost and transportation cost over out-of-sample scenarios without seasonality, which have been shown in Figure 6.
5. The files [Ave_total_cost_robustness_phi_0.xlsx](results/Out_of_Sample_Robustness/Ave_total_cost_robustness_phi_0.xlsx),...,[Ave_total_cost_robustness_phi_5.xlsx](results/Out_of_Sample_Robustness/Ave_total_cost_robustness_phi_5.xlsx) include the averaged out-of-sample total cost for the DRO, S-DRO and SAA solutions across 100 tests with different proportion of the worst-case distribution \phi \in {0.0, 0.1, 0.3, 0.5}. Similarly, the files [Ave_transportation_cost_robustness_phi_0.xlsx](results/Out_of_Sample_Robustness/Ave_transportation_cost_robustness_phi_0.xlsx),...,[Ave_transportation_cost_robustness_phi_5.xlsx](results/Out_of_Sample_Robustness/Ave_transportation_cost_robustness_phi_5.xlsx) and the files [Ave_shortage_cost_robustness_phi_0.xlsx](results/Out_of_Sample_Robustness/Ave_shortage_cost_robustness_phi_0.xlsx),...,[Ave_shortage_cost_robustness_phi_5.xlsx](results/Out_of_Sample_Robustness/Ave_shortage_cost_robustness_phi_5.xlsx) record the averaged out-of-sample transportation cost and averaged
out-of-sample shortage cost under three solutions. All the results in the folder **Out_of_Sample_Robustness** have been shown in Figure 7.
6. The file [Result_sensitivity_sigma.xlsx](results/Sensitivity_Analysis/Result_sensitivity_sigma.xlsx) in the folder **Sensitivity_Analysis** records the changes of worst-case expected transportation cost and its theoretical upper bounds in different states with respect to the perturbation of the ambiguity parameter \sigma_k, which has been plotted in Figure 8. Similarly, the files [Result_sensitivity_d_plus.xlsx](results/Sensitivity_Analysis/Result_sensitivity_d_plus.xlsx) and [Result_sensitivity_d_minus.xlsx](results/Sensitivity_Analysis/Result_sensitivity_d_minus.xlsx) include the changes of worst-case expected transportation cost and its theoretical upper bounds with respect to the ambiguity parameters d^+_k and d^-_k, which are respectively shown in Figure 9 and Table 3.
7. The file [CPU_time_uncapacitated_problem.xlsx](results/CPU_Time_Uncapacitated_Problem/CPU_time_uncapacitated_problem.xlsx) in the folder **CPU_Time_Uncapacitated_Problem** records the computational time and expected total cost of the uncapacitated problem with different sizes of instances, which has been reported in Table 4.
8. The files [Result_S_NBD_I_10_J_10_K_1.xlsx](results/Algorithm_Performance/Capacitated_I_10_J_10/Result_S_NBD_I_10_J_10_K_1.xlsx),...,[Result_S_NBD_I_20_J_50_K_5.xlsx](results/Algorithm_Performance/Capacitated_I_20_J_50/Result_S_NBD_I_20_J_50_K_5.xlsx) record the computational time and gap of each iteration of the S-NBD algorithm for solving the capacitated problem with different instance sizes, which has been shown in Table 5. Furthermore, the files [Result_BD_I_10_J_10_K_1.xlsx](results/Algorithm_Performance/Capacitated_I_10_J_10/Result_BD_I_10_J_10_K_1.xlsx),...,[Result_BD_I_20_J_50_K_5.xlsx](results/Algorithm_Performance/Capacitated_I_20_J_50/Result_BD_I_20_J_50_K_5.xlsx) include the computational time and gap of the BD algorithm for solving the capacitated problem with different sizes of instance, which has also been recorded in Table 5 and plotted in Figure 10 and 11.


## Replicating

To replicate the results in our paper, the users should put all the files under the same foldername in one folder and run the main programs which have been mentioned in the Content. For instance, to obtain the results in Table 1 and Table 2 (the file [Result_DRO_SDRO_SAA.xlsx](/results/Solution_DRO_SDRO_SAA/Result_DRO_SDRO_SAA.xlsx) in the folder **Solution_DRO_SDRO_SAA**), one should put the files [Parameters.xslx](data/Solution_DRO_SDRO_SAA/Parameters.xlsx), [Parameters_DRO.xlsx](data/Parameters_DRO.xlsx), [Parameters_SAA](data/Parameters_SAA.xlsx), [Parameters_SDRO](data/Parameters_SDRO.xlsx) and the files [Solution_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Solution_DRO_SDRO_SAA.py), [Func_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Func_DRO_SDRO_SAA.py) into the same folder, then run the main program [Solution_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Solution_DRO_SDRO_SAA.py) in Python.       


