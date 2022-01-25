# Improvement of P&R facility and services in metropolitain areas of Tennessee

************************************************* OPT_TAZ python files ************************************************* 


************************************************* Install Python ************************************************* 
## Install Python 3
## Create an environment and isntall the following packages on that environment:
       numpy 
       pandas
       Pyomo
## To run the optimization model using "Pyomo", it is needed to install a solver. It is suggested to use "gurobi" solver as it is faster, but one can use free solvers such as "glpk" or "cbc" (Note that these two free solvers are much slower). 
       Install solvers "gurobi", "glpk" or "cbc"


************************************************* Model Inputs *************************************************
## "data_P&R.csv" includes individual specific attributes, SOV, HOV and transit times to the CBD area and SOV time to P&R stations for each individual (used for all models).
## "data_P&R_dis.csv" includes all the information of "data_P&R.csv" file plus distances from origins to CBD area and to P&R stations (used for 'ER' objective function and TOD vs P&R analysis).
## "MNL_bal_Coef_output.csv" includes estimated coefficients obtained by "MNL_Balanced.py" (used for all models) by "report_data.csv". 
## "Transit_time_P&R_CBD.csv" includes transit time from P&R stations to the CBD area (used for all models).


************************************************* OPT_TAZ_Approach_2.py *************************************************
## This code analyzes Approach 2 to estimate the coefficients of variables in the P&R utility function in the context of the coefficients of attributes for the existing modes.
## This code requests for two inputs from users:
      Alpha: Please select a value between 0 and 1. This is the weight of HOV coefficients in Approach 2
      Solver: Please select the solver you installed on the environment ('gurobi', 'glpk' or 'cbc')
## This code outputs a ".csv" file including the usage of all modes (SOV, HOV, transit and P&Rs) in each county based on the alpha entered.
## This code also displays the summary of results: The total usage of each mode and their percentage usages


************************************************* OPT_TAZ_Abs_Con_placement_P&R.py *************************************************
## This code analyzes the MILP model in the absence of constraints on placement of P&Rs. 
## This model finds the optimal placement of P&Rs when a specific number of candidate P&Rs should be chosen. 
## This code requests for three inputs from users:
      The number of candidate P&Rs should be established: Please enter a value between 1 and 11
      Solver: Please select the solver you installed on the environment ('gurobi', 'glpk' or 'cbc')
      Objective function: Please specify the objective function in the optimization model (You can enter 'UM' if you wish to maximize the utilization or enter 'ER' if you wish to minimize the emission).
## This code outputs a ".csv" file including the usage of all modes (SOV, HOV, transit and P&Rs) in each county.
## This code also displays the summary of results: The total usage of each mode and their percentage usages.
## You can do the analysis based on two different objective functions.


************************************************* OPT_TAZ_Acc_Con_placement_P&R.py *************************************************
## This code analyzes the MILP model When accounting for constraints on placement of P&Rs
## We conducted experiments while incorporating constraints on the placement of P&Rs
## This code requests for two user inputs:
      Solver: Please select the solver you installed on the environment ('gurobi', 'glpk' or 'cbc')
      Objective function: Please specify the objective function in the optimization model (You can enter 'UM' if you wish to maximize the utilization or enter 'ER' if you wish to minimize the emission).
## This code outputs a ".csv" file including the usage of all modes (SOV, HOV, transit and P&Rs) in each county. 
## This code also displays the summary of results: The total usage of each mode and their percentage usages.
## You can do the analysis based on two different objective functions.


************************************************* OPT_TAZ_Sensitivity_Analysis.py *************************************************
## This code provides sensitivity analysis regarding Travel Time, Traffic Flow, and Population Growth
## This code provides analysis for all 11 scenario considered in the report
## This code requests for Four inputs from users:
      The number of candidate P&Rs should be established: Please enter a value between 1 and 11
      Solver: Please select the solver you installed on the environment ('gurobi', 'glpk' or 'cbc')
      Objective function: Please specify the objective function in the optimization model (You can enter 'UM' if you wish to maximize the utilization or enter 'ER' if you wish to minimize the emission).
      Scenario: Please enter the scenario number from 1 to 11
## This code outputs a ".csv" file including the usage of all modes (SOV, HOV, transit and P&Rs) in each county.
## This code also displays the summary of results: The total usage of each mode and their percentage usages.
## You can do the analysis based on two different objective functions.


************************************************* TOD_vs_P&R.py *************************************************
## This code compares TODs and P&Rs in the City of Nashville. 
## This code requests for three inputs from users:
      The number of candidate P&Rs should be established: Please enter a value between 1 and 11
      Solver: Please select the solver you installed on the environment ('gurobi', 'glpk' or 'cbc')
      Objective function: Please specify the objective function in the optimization model (You can enter 'UM' if you wish to maximize the utilization or enter 'ER' if you wish to minimize the emission).
## This code outputs ".csv" files including all parameters to compare TOD versus P&R such as RHP, R and A.
      You need to look at '_RT.csv' and '_UH.csv' for the final comparison. 
      '_RT.csv' is the Minimum average VKT reduction per household needed for the TOD to meet the target VKT reduction per P&R land hectare under different levels of residential density
      '_UH.csv' is the Minimum housing units per hectare needed for the TOD to meet the target VKT reduction per P&R land hectare under different levels of VKT reduction per household
## You can do the analysis based on two different objective functions.


