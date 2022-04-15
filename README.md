[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Robust Stochastic Facility Location: Sensitivity Analysis and Exact Solution

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper "Robust Stochastic Facility Location: Sensitivity Analysis and Exact Solution" by T. Liu, F. Saldanha-da-Gama, S. Wang and Y. Mao. 

## Cite

To cite this software, please cite this repository, using the following DOI.

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

1. In folder **Solution_DRO_SDRO_SAA**, the file [Parameters.xslx](data/Solution_DRO_SDRO_SAA/Parameters.xlsx) includes the fixed cost, capacity, transportation cost and penalty cost, which are used in the expertiments. Furthermore, the files [Parameters_DRO.xlsx](data/Parameters_DRO.xlsx), [Parameters_SAA](data/Parameters_SAA.xlsx) and [Parameters_SDRO](data/Parameters_SDRO.xlsx) include the parameters of the DRO model, SAA model and S-DRO model proposed in our paper, respectively.  
2. The file [d_sample_set_seasonality.xlsx](data/Out_of_Sample_seasonality/d_sample_set_seasonality.xlsx) in folder **Out_of_Sample_seasonality** contains 100 sets of randomly generated demand samples with seasonal information. 
3. The file [d_sample_set_nonseasonality.xlsx](data/Out_of_Sample_nonseasonality/d_sample_set_nonseasonality.xlsx) in folder **Out_of_Sample_nonseasonality** contains 100 sets of randomly generated deamand samples without seasonal information.
4. The file [demand_observation_2019.xlsx](data/Real_Case_2019/demand_observation_2019.xlsx) contains the true observation of the demand in each quarter of 2019.
5. In folder **Out_of_Sample_Robustness**, the file [d_sample_set_phi_0.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_0.xlsx),...,[d_sample_set_phi_5.xlsx](data/Out_of_Sample_Robustness/d_sample_set_phi_5.xlsx) are 100 sets of randomly generated demand samples with different proportion of the worst-case distribution \phi \in {0.0, 0.1, 0.3, 0.5}.
6. In folder **CPU_Time_Uncapacitated_Problem**, the files [Uncapacitated_I_10_J_10_K_1.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_10_J_10_K_1.npy),..., [Uncapacitated_I_20_J_50_K_5.npy](data/CPU_Time_Uncapacitated_Problem/Uncapacitated_I_20_J_50_K_5.npy) contain the parameters of the uncapacitated S-DRO model with different number of the potential location sites I, customer sites J and states K. 

### Script files 

The code folders include **CoASLTP**, **CoPathsolver** and **Bargaining**.
1. The code in the folder **CoASLTP** is for comparing the proposed interior-point difierentiable path-following method (**IPM**) and the **ASLTP**, where the file [ycsgse.m](CoASLTP/ycsgse.m) is the main program of the **IPM** and [dltpsgse.m](CoASLTP/dltpsgse.m) is the main program of the **ASLTP**. The code in this folder has been used in Section 4.1.
2. The code in the folder **CoPathsolver** is for comparing the proposed **IPM** and the **path solver**, where the file [trysg.m](CoPathsolver/trysg.m) is the main program. The code in this folder has been used in Sections 4.2 and 4.3.
3. The folder **Bargaining** includes the code for computing a solution to the bargaining model, which has been presented in Section 4.4.

### Results files

The results folders **comp-ASLTP**, **comp-pathsolver** and **morecomplicated** record the computational results for all numerical examples used in the paper.
1. The files [A0.txt](comp-ASLTP/A0.txt), [A1.txt](comp-ASLTP/A1.txt), [A2.txt](comp-ASLTP/A2.txt), [A3.txt](comp-ASLTP/A3.txt) in the folder **comp-ASLTP** consist of the computational time and number of iterations of **IPM** and **ASLTP** for solving various stochastic games. We summarize these comparison results in the file [comp-sltp.xlsx](comp-ASLTP/comp-sltp.xlsx). The average computational time has been reported in Table 1 of the manuscript. 
2. The file [comp-path.txt](comp-pathsolver/comp-path.txt) in the folder **comp-pathsolver** includes the comparison results between the proposed **IPM** and the **path solver**. We summarize these results in an excel sheet named [comp-path.xlsx](comp-pathsolver/comp-path.xlsx), which is exactly Table 2 in the manuscript. Additionally, the file [test-success-rate.txt](comp-pathsolver/test-success-rate.txt) consists of the results of the path solver for stochastic games with different scales and the success rate among 100 randomly generated examples for each case.
3. The files [rdata0.txt](morecomplicated/rdata0.txt),...,[rdata11.txt](morecomplicated/rdata11.txt) in the folder **morecomplicated** record the computational time of **IPM** for all large-scale stochastic games. We summarize these instance results in [more-complicated.xlsx](morecomplicated/more-complicated.xlsx). The average computation time for each case has been reported in Table 3 of the paper.

We finally illustrate how to implement the code and associate the code files with the numerical results (e.g., tables and figures) presented in the paper.
1. By running the files [exm1.m](CoASLTP/exm1.m), [exm2.m](CoASLTP/exm2.m), [exm3.m](CoASLTP/exm3.m),  [exm4.m](CoASLTP/exm4.m), [exm5.m](CoASLTP/exm5.m) in the folder **CoASLTP**, one can obtain the computational results for the five fundamental examples in Section 4.1.
5. By running the file [inputs225.m](CoASLTP/inputs225.m) in the folder **CoASLTP**, one can get a stationary equilibrium for a stochastic game with two players, two states and five actions for each player in each state. The computational costs of the **IPM** and **ASLTP** for solving this instance are obtained as well. By repeatedly running [inputs225.m](CoASLTP/inputs225.m) for ten times, one can obtain the average computational time for both methods, which is shown in the first row of Table 1. Through changing the parameters *n*; *d*; *m*; *pd0* in [inputs225.m](CoASLTP/inputs225.m), we can attain various instances. The average computational time of IPM for solving these stochastic games are shown in Table 1 and Table 3.
6. By implementing the file  [se225.m](CoPathsolver/se225.m) in the folder **CoPathsolver**, one can get the comparison results between the proposed **IPM** and the **path solver** for computing a stationary equilibrium in a randomly generated stochastic game with two players, two states and five actions. Similarly, by changing the parameters *n*; *d*;*m*, we attain various stochastic games with difierent scales. The comparison results are included in Table 2. By running the file [r1.m](CoPathsolver/r1.m), one may obtain the success rates of the two methods for 100 randomly generated stochastic games, which are recorded in Figure 5.
7. By implementing the file [bargaining.m](Bargaining/bargaining.m) in the folder **Bargaining**, one can get Figure 6, which shows a solution to the presented bargaining model.


## Building

In Linux, to build the version that multiplies all elements of a vector by a
constant (used to obtain the results in [Figure 1](results/mult-test.png) in the
paper), stepping K elements at a time, execute the following commands.

```
make mult
```

Alternatively, to build the version that sums the elements of a vector (used
to obtain the results [Figure 2](results/sum-test.png) in the paper), stepping K
elements at a time, do the following.

```
make clean
make sum
```

Be sure to make clean before building a different version of the code.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

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

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
