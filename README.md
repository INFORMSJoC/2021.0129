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
1. In the folder **Solution_DRO_SDRO_SAA**, the file [Parameters.xslx](data/Solution_DRO_SDRO_SAA/Parameters.xlsx) includes the fixed cost, capacity, transportation cost and penalty cost, which are used in the expertiments. Furthermore, the files [Parameters_DRO.xlsx](data/Parameters_DRO.xlsx), [Parameters_SAA](data/Parameters_SAA.xlsx) and [Parameters_SDRO](data/Parameters_SDRO.xlsx) include the parameters of the DRO model, SAA model and S-DRO model proposed in our paper, respectively.  
2. The file [d_sample_set_seasonality.xlsx](data/Out_of_Sample_seasonality/d_sample_set_seasonality.xlsx) in the folder **Out_of_Sample_seasonality** contains 100 sets of randomly generated demand samples with seasonal information. 
3. The file [d_sample_set_nonseasonality.xlsx](data/Out_of_Sample_nonseasonality/d_sample_set_nonseasonality.xlsx) in the folder **Out_of_Sample_nonseasonality** contains 100 sets of randomly generated deamand samples without seasonal information.
4. The file [demand_observation_2019.xlsx](data/Real_Case_2019/demand_observation_2019.xlsx) in the folder **Real_Case_2019** contains the true observation of the demand in each quarter of 2019.
5. In the folder **Out_of_Sample_Robustness**, the file [d_sample_set_phi_0.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_0.xlsx),...,[d_sample_set_phi_5.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_5.xlsx) contain 100 sets of randomly generated demand samples with different proportion of the worst-case distribution \phi \in {0.0, 0.1, 0.3, 0.5}, respectively.
6. In the folder **CPU_Time_Uncapacitated_Problem**, the files [Uncapacitated_I_10_J_10_K_1.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_10_J_10_K_1.npy),..., [Uncapacitated_I_20_J_50_K_5.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_20_J_50_K_5.npy) contain the parameters of the uncapacitated S-DRO model with the different number of the potential location sites I, customer sites J and states K. 
7. The files [Capacitated_I_10_J_10_K_1.npy](data/Algorithm_Performance/Capacitated_I_10_J_10/Capacitated_I_10_J_10_K_1.npy),...,[Capacitated_I_20_J_50_K_5.npy](data/Algorithm_Performance/Capacitated_I_20_J_50/Capacitated_I_20_J_50_K_5.npy) in the folder **Algorithm_Performance** contain the parameters of the capacitated S-DRO model with the different number of the potential location sites I, customer sites J and states K.

### Code files 

The folder **scripts** includes all the codes used in our experiments.
1. The code files in the folder **Solution_DRO_SDRO_SAA** are for obtaining the solution profiles for the DRO, S-DRO and SAA models, where the file [Solution_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Solution_DRO_SDRO_SAA.py) is the main program and [Func_DRO_SDRO_SAA.py](scripts/Solution_DRO_SDRO_SAA/Func_DRO_SDRO_SAA.py) contains all the functions used in the main program. The codes have been used in Section 6.1 (Value of state-wise distributional information) and 6.2 (Value of robustness) in our paper.
2. The code file [Out_of_Sample_seasonality.py](scripts/Out_of_Sample_seasonality/Out_of_Sample_seasonality.py) in the folder **Out_of_Sample_seasonality** is for evaluating the performance of the S-DRO solution compared with the DRO solution using the out-of-sample tests with the seasonal information, which has been used in Section 6.1.
3. The code file [Real_Case_2019.py](scripts/Real_Case_2019/Real_Case_2019.py) in the folder **Real_Case_2019** is for evaluating the performance of the S-DRO and DRO solutions using the true demand observations in 2019, which has been used in Section 6.1.
4. The code file [Out_of_Sample_nonseasonality.py](scripts/Out_of_Sample_nonseasonality/Out_of_Sample_nonseasonality.py) in the folder **Out_of_Sample_nonseasonality** is for testing the performance of both S-DRO and DRO solutions under the out-of-sample scenarios without seasonality, which has been used in Section 6.1.
5. The code file [Out_of_Sample_Robustness.py](scripts/Out_of_Sample_Robustness/Out_of_Sample_Robustness.py) in the folder **Out_of_Sample_Robustness** is for  evaluating the value of robustness of the proposed S-DRO model and the DRO model by comparing their out-of-sample performance with the SAA counterpart, which has been used in Section 6.2.
6. The code files in the folder **Sensitivity_Analysis** is for evaluating the changes of worst-case expected transportation cost in different states with respect to the perturbation of the ambiguity parameters \sigma_k, d^+_k and d^-_k, where the files [Sensitivity_sigma.py](scripts/Sensitivity_Analysis/Sensitivity_sigma.py), [Sensitivity_d_plus.py](scripts/Sensitivity_Analysis/Sensitivity_d_plus.py) and [Sensitivity_d_minus.py](scripts/Sensitivity_Analysis/Sensitivity_d_minus.py) are the main program for different parameters and the file [Func_Sensitivity.py](scripts/Sensitivity_Analysis/Func_Sensitivity.py) contains all the functions used in the main programs. The codes have been used in Section 6.3 (Sensitivity analysis of ambiguity set parameters) in our paper.
7. The code files in the folder **CPU_Time_Uncapacitated_Problem** are for evaluating the computational performance of solving the uncapacitated S-DRO model exactly in the reformulation of mixed-interger second-order cone program, where the file [CPU_Time_Uncapacitated_Problem.py](scripts/CPU_Time_Uncapacitated_Problem/CPU_Time_Uncapacitated_Problem.py) is the main program and the file [Func_CPU_Time_Uncapacitated_Problem.py](scripts/CPU_Time_Uncapacitated_Problem/Func_CPU_Time_Uncapacitated_Problem.py) contains all the functions used in the main program. The codes in this folder have been used in Section 6.4 (Performance of the exact solution approach) in our paper.
8. The code files in the folder **Algorithm_Performance** is for comparing the subgradient-based Nested Benders decomposition algorithm (S-NBD) and the Benders decomposition approach (BD) under different instances. The files [S_NBD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/S_NBD_I_10_J_10.py),...,[S_NBD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/S_NBD_I_20_J_50.py) are the main programs of the S-NBD under different instances and the files [Func_S_NBD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/Func_S_NBD_I_10_J_10.py),...,[Func_S_NBD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/Func_S_NBD_I_20_J_50.py) includes all the functions used in the main programs. Similarly, the files [BD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/BD_I_10_J_10.py),...,[BD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/BD_I_20_J_50.py) are the main programs of BD under different instances and the files [Func_BD_I_10_J_10.py](scripts/Algorithm_Performance/Capacitated_I_10_J_10/Func_BD_I_10_J_10.py),...,[Func_BD_I_20_J_50.py](scripts/Algorithm_Performance/Capacitated_I_20_J_50/Func_BD_I_20_J_50.py) contains all the functions used in the main program. The codes have been used in Section 6.4. 

### Results files

The folder **results** contains the results for all numerical experiments used in our paper.
1. The file [Result_DRO_SDRO_SAA.xlsx](/results/Solution_DRO_SDRO_SAA/Result_DRO_SDRO_SAA.xlsx) in the folder **Solution_DRO_SDRO_SAA** includes the solution profiles for the DRO, S-DRO and SAA models, which is exactly Table 1 and Table 2 in our paper.
2. The file [out_of_sample_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_total_cost_seasonality.xlsx) in the folder **Out_of_Sample_seasonality** includes the out-of-sample total cost of both S-DRO and DRO location designs across 100 tests with seasonal informatiom. Similarly, the files  [out_of_sample_conservativeness_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_total_cost_seasonality.xlsx) and [out_of_sample_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/out_of_sample_transportation_cost_seasonality.xlsx) contain the out-of-sample conservativeness and out-of-sample transportation cost, respectively. Furthermore, the files [Ave_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_total_cost_seasonality.xlsx), [Ave_conservativeness_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_conservativeness_seasonality.xlsx) and [Ave_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Ave_transportation_cost_seasonality.xlsx) include the averaged out-of-sample total cost, conservativeness value and transportation cost of each test, respectively. The above results have been recorded in Figure 3 and Figure 4. Finally, the file [Diff_Ave_total_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Diff_Ave_total_cost_seasonality.xlsx) includes the difference between the average total cost corresponding to
the DRO solution and that for the S-DRO solution, as well as the file [Diff_Ave_transportation_cost_seasonality.xlsx](results/Out_of_Sample_seasonality/Diff_Ave_transportation_cost_seasonality.xlsx) contains the the differences between the average transportation costs for the DRO and the S-DRO solutions. The results have been presented in Figure 6.
3. The file [Result_real_case_2019.xlsx](results/Real_Case_2019/Result_real_case_2019.xlsx) in the folder **Real_Case_2019** records the total costs and the associate out-of-sample conservativeness values in each quarter of 2019 under both S-DRO and DRO solutions, which has been plotted in Figure 5.
4. 
5. The files [A0.txt](comp-ASLTP/A0.txt), [A1.txt](comp-ASLTP/A1.txt), [A2.txt](comp-ASLTP/A2.txt), [A3.txt](comp-ASLTP/A3.txt) in the folder **comp-ASLTP** consist of the computational time and number of iterations of **IPM** and **ASLTP** for solving various stochastic games. We summarize these comparison results in the file [comp-sltp.xlsx](comp-ASLTP/comp-sltp.xlsx). The average computational time has been reported in Table 1 of the manuscript. 
6. The file [comp-path.txt](comp-pathsolver/comp-path.txt) in the folder **comp-pathsolver** includes the comparison results between the proposed **IPM** and the **path solver**. We summarize these results in an excel sheet named [comp-path.xlsx](comp-pathsolver/comp-path.xlsx), which is exactly Table 2 in the manuscript. Additionally, the file [test-success-rate.txt](comp-pathsolver/test-success-rate.txt) consists of the results of the path solver for stochastic games with different scales and the success rate among 100 randomly generated examples for each case.
7. The files [rdata0.txt](morecomplicated/rdata0.txt),...,[rdata11.txt](morecomplicated/rdata11.txt) in the folder **morecomplicated** record the computational time of **IPM** for all large-scale stochastic games. We summarize these instance results in [more-complicated.xlsx](morecomplicated/more-complicated.xlsx). The average computation time for each case has been reported in Table 3 of the paper.

We finally illustrate how to implement the code and associate the code files with the numerical results (e.g., tables and figures) presented in the paper.
1. By running the files [exm1.m](CoASLTP/exm1.m), [exm2.m](CoASLTP/exm2.m), [exm3.m](CoASLTP/exm3.m),  [exm4.m](CoASLTP/exm4.m), [exm5.m](CoASLTP/exm5.m) in the folder **CoASLTP**, one can obtain the computational results for the five fundamental examples in Section 4.1.
5. By running the file [inputs225.m](CoASLTP/inputs225.m) in the folder **CoASLTP**, one can get a stationary equilibrium for a stochastic game with two players, two states and five actions for each player in each state. The computational costs of the **IPM** and **ASLTP** for solving this instance are obtained as well. By repeatedly running [inputs225.m](CoASLTP/inputs225.m) for ten times, one can obtain the average computational time for both methods, which is shown in the first row of Table 1. Through changing the parameters *n*; *d*; *m*; *pd0* in [inputs225.m](CoASLTP/inputs225.m), we can attain various instances. The average computational time of IPM for solving these stochastic games are shown in Table 1 and Table 3.
6. By implementing the file  [se225.m](CoPathsolver/se225.m) in the folder **CoPathsolver**, one can get the comparison results between the proposed **IPM** and the **path solver** for computing a stationary equilibrium in a randomly generated stochastic game with two players, two states and five actions. Similarly, by changing the parameters *n*; *d*;*m*, we attain various stochastic games with difierent scales. The comparison results are included in Table 2. By running the file [r1.m](CoPathsolver/r1.m), one may obtain the success rates of the two methods for 100 randomly generated stochastic games, which are recorded in Figure 5.
7. By implementing the file [bargaining.m](Bargaining/bargaining.m) in the folder **Bargaining**, one can get Figure 6, which shows a solution to the presented bargaining model.


## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```
