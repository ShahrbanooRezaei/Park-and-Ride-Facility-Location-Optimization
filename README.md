# Improvement of P&R facility and sercices in metropolitain areas of Tennessee

#### There are four main code categories in this folder:


(1) Codes related to extracting information needed for this report. 
    ## "_household.tsv", "_person.tsv" and "_trip.tsv" are three datasets we recieved from TDOT. We needed to mix these datasets and extract information required for this project.
    ## These three datasets along with "QGISdata.csv" are imported into codes "Data_extraction.py" and "Data_extraction_dis.py" to extract "data_P&R.csv" and "data_P&R_dis.csv" which are used for the MNL model and optimization model.


(2) Codes related to implementing the multinomial logit model (MNL). These codes are general and can be implemented on any datasets. The code requirements and data structured are explained in "ReadMe" file in the folder.
    ## We provided two codes "MNL_Balanced.py" and "MNL.py" for implementing the MNL model calibrating on both balanced and imbalanced datasets, respectively. 
    ## In order to implement these codes on the data of this report, you can use "report_data.csv" which is extracted from "data_P&R.csv" file.
       # You are asked to enter some inputs when runnig these codes. You must know the structure of data before the implementation. Here is the help!
       # For "report_data.csv": 
                               1. "Please enter the name of data file like 'data'": Please enter "report_data"
                               2. "Please enter the number of individual specific variables": Please enter 22
                               3. "Please enter the number of alternative specific variables": Please enter  1
                               4. "Please enter the base mode": Please select 1. It means that you select SOV (1) as base mode. (you can choose different values for this but if you want to repeat the report outcoms you must enter 1)
                               5. "Please enter 1 if you want the alternative variable 1 to be global, otherwise please enter 0". Please select 0. (you can choose different values for this but if you want to repeat the report outcoms you must enter 0)
    ## We also provided a synthetic dataset called "Syn_data_P&R.csv" that has more alternative specific variables and has P&R as selected modes by individuals to show the generalizability of these codes on any dataset.
       # You are asked to enter some inputs when runnig these codes. You must know the structure of data before the implementation. Here is the help!
       # For "Syn_data_P&R.csv": 
                               1. "Please enter the name of data file like 'data'": Please enter "Syn_data_P&R"
                               2. "Please enter the number of individual specific variables": Please enter 12
                               3. "Please enter the number of alternative specific variables": Please enter  2
                               4. "Please enter the base mode": You can select a value from {1,2,3}. It means that you select SOV (1), HOV(2) or transit (3) as the base mode.
                               5. "Please enter 1 if you want the alternative variable i to be global, otherwise please enter 0". you can select either 0 or 1 for each variable. 
       # Please remember your inputs for part 4 and 5. You will need it for code in (4)
    ## The outputs of these codes "MNL_bal_Coef_output.csv" or "MNL_Coef_output.csv" is used for part (3) and (4) codes.


(3) codes related to the results of different parts in the report.
    ## Codes in this part use "MNL_bal_Coef_output.csv" obtained by "MNL_Balanced.py" by applying "report_data.csv".
    ## These codes also use "data_P&R.csv", "data_P&R_dis.csv" and "Transit_time_P&R_CBD.csv" files
   

(4) This code is the generalized code for the optimization model that can be implemented on any dataset.  
    ## This code finds the optimal placement of P&Rs when a specific number of candidate P&Rs should be chosen from a set of candidate P&Rs. 
    ## The dataset has to have a specific structure explained in ReadMe file in the folder. 
    ## We used a synthetic dataset called "Syn_data_P&R.csv" that has more alternative specific variables and has P&R as selected modes by individuals to show the generalizability of this code on any dataset.   
       # This dataset has three P&Rs and you can use your set of existing and candidate from these P&Rs. 
       # You are asked to enter some inputs when runnig the code. 
       # For "Syn_data_P&R.csv": 
                               1. "Please enter the number of individual specific variables": Please enter 12
                               2. "Please enter the number of alternative specific variables": Please enter  2
                               3. "Please enter the base mode": Please enter the value you entered in part (2) under question 4.
                               4. "Please enter 1 if you want the alternative variable i to be global, otherwise please enter 0": Please enter the value you entered in part (2) under question 5.
       
                               5. "Please identify the solver. You can enter gurobi, cbc or glpk": Please identify the one you installed. 
                               6. "Please enter 'Yes' if you want to use coefficients obtained from 'MNL_Balanced.py' or enter 'No' otherwise": You can either choose 'Yes' or 'No', depending on your previous implementation on part (2).
                               7. "Please enter the number of existing P&Rs": There are three park and rides so you can either choose 1 or 2.
                               8. "Please enter the number of candidate P&Rs": Based on your previous answer, you can enter 1 or 2.
                               9. "Please enter an integer value to especify the number of candidate P&Rs should be established": Based on previous answer, you can choose 0, 1 or 2.
                               10. "Please enter 'UM' if you want to optimize based on the utilization maximization or enter 'ER' if you want to optimize based on the emission reduction"
                               11. If you select 'ER', you need to:
                                   1. "Please enter SOV mode number if it presents in 'mode' column otherwise please enter 0": Please enter 1, as SOV represented as 1 is presented in data.
                                   2. "Please enter HOV mode number if it presents in 'mode' column otherwise please enter 0": Please enter 2, as HOV represented as 2 is presented in data.
                                   3. "Please enter Transit mode number if it presents in 'mode' column otherwise please enter 0": Please enter 3, as transit represented as 3 is presented in data.
    
    ## You also can use your own dataset if it has the structure. But before use your own dataset in the optimization model, you should implement the MNL model on your dataset to get the estimated coefficients and use these coefficients as inputs to "OPT_General_code.py" code.

