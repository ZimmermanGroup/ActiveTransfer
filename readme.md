# Predicting Reaction Conditions from Limited Data through Active Transfer Learning
Code for the paper.

## Contents
1. array_joblibs  
Includes pickled arrays that are used as inputs throughout the notebooks.
Particularly, additional arrays that were subject to ATL experiments in the Supporting Information is provided
in `arrays_for_additional_ATL`.

2. prep_full_rxn_arrays.ipynb  
Prepares numpy arrays from all datapoints of `rxn_db.sql`.

3. transfer_between_common_rxn_conditions.ipynb  
Code to generate results presented in Figure 3.

4. model_complexity_and_transfer_performance.ipynb  
Code to generate results presented in Figure 4.

5. ActiveTransfer.ipynb  
Code to generate results presented in Figures 5~7.
Compares various active transfer learning strategies and analysis of the 'target tree growth' strategy.
* AL.py : 
Code for active learning. 
* compare_ATL_strategies.py : 
Plots the performance of each ATL strategy for finding desired reactions.
* passive_model_perf.py : 
Conducts iterative reaction selection suggested by the source model without any updates.
* update_combined_data.py : 
Conducts active transfer learning based on combined source and collected target data.
* eval_adaptability.py : 
Compares how models updated each iteration perform on the data in hand - source versus collected target.
* analyze_target_tree_growth.py : 
Analyzes the models of target tree growth strategy and the importance of model simplicity.

6. xyzfiles  
Includes xyz coordinates of all compounds used in this study.

7. requirements.txt  
Lists the version used of core libraries used in this work.

## Citing
If you find the code within this repo useful, please consider citing :  
`Shim, E.; Kammeraad, J. A.; Xu, Z.; Tewari, A.; Cernak, T.; Zimmerman, P. M. Predicting Reaction Conditions from Limited Data through Active Transfer Learning, [Journal Name], Year, Issue, Pages`