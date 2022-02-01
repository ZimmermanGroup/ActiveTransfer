import copy
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from AL import *
import matplotlib.pyplot as plt
import seaborn as sns


def compare_ATL_strategies(
    source_list, source_model_list, source_desc_list, source_y_list,
    target_desc, target_id, target_y,
    num_rxns_per_batch=3, new_max_depth=1,
    num_trees_to_add=3, target_weight_in_combined=5,
    print_progress=False, strategies=["Passive","Combined Data", "Add 3 Trees", "No Source"]
):
    """ Compares the performances of the following ATL strategies.
    • Target tree growth  • Re-train on combined data  • Passive modeling (no source model update)
    • No source (modeling only based on target data)
    
    Parameters
    ----------
    source_list : list of str
        Source nucleophile names
    source_model_list : lists of list of classifiers
        Each embedded list has all classifiers of a source domain.
    source_desc_list, source_y_list : list of np.2d/1darrays
        Contains source descriptor / yield label arrays.
    target_desc, target_id, target_y : np.2d/2d/1darrays
        Arrays of target descriptor, id, yield labels.

    Returns
    -------
    adding_tree_system : dict
        Contains all information of ATL simulations conducted.
    adding_tree_models : dict
        Contains all models updated and reactions collected along the ATL simulations.
    """
    adding_tree_system = {}
    adding_tree_models = {}
    num_desired = len(target_y) - sum(target_y)
    len_iter_result = len(target_y)//num_rxns_per_batch + 2

    for i, source in enumerate(tqdm(source_list)):
        # Preparing Random Models
        model_list = source_model_list[i]
        scenario = f"{source}_to_heterocycle"
        adding_tree_system.update({scenario:{
                    "num_rxns_conducted":[],
                    "num_rxns_found":[],
                    "Strategy":[]
            }})
        adding_tree_models.update({scenario:{
                    "models":[],
                    "arrays":[]
            }})
        
        for j, model in enumerate(model_list):
            for strategy in strategies: # "No Source - 5 Trees", 
                if strategy in ["Passive", "Add 3 Trees", "Combined Data"]:
                    #print(f"    Add {num_trees} trees, random_state={42+k}")
                    model_to_use = copy.deepcopy(model)
                    #print(f"        Starting with {len(model.estimators_)} Trees")
                    if strategy=="Add 3 Trees":
                        rxns_collected_per_batch, confidence_selected_rxns,\
                        model_by_iter, num_found_by_batch = explore_target_in_batches(
                            model_to_use, source_desc_list[i], source_y_list[i],
                            target_desc, target_id, target_y,
                                    "confidence", "add_collected",
                                    num_rxns_per_batch = num_rxns_per_batch,
                                    num_trees_to_add=num_trees_to_add, new_max_depth=new_max_depth,
                                    enough_found=num_desired,
                                    print_progress=print_progress,
                                    random_state=42+j
                                )
                    elif strategy=="Passive" : 
                        rxns_collected_per_batch, confidence_selected_rxns,\
                        model_by_iter, num_found_by_batch = explore_target_in_batches(
                            model, source_desc_list[i], source_y_list[i],
                            target_desc, target_id, target_y,
                            "confidence", "none",
                            num_rxns_per_batch=num_rxns_per_batch,
                            weight_factor=0,
                            enough_found=num_desired,
                                    print_progress=print_progress,
                                    random_state=42+j
                                )
                    elif strategy=="Combined Data":
                        rxns_collected_per_batch, confidence_selected_rxns,\
                        model_by_iter, num_found_by_batch = explore_target_in_batches(
                            model, source_desc_list[i], source_y_list[i],
                            target_desc, target_id, target_y,
                            "confidence", "new",
                            num_rxns_per_batch=num_rxns_per_batch,
                                    num_trees_to_add=num_trees_to_add,
                            new_max_depth=new_max_depth,
                            weight_factor=target_weight_in_combined,
                            enough_found=num_desired,
                                    print_progress=print_progress,
                                    random_state=42+j
                                )
                else :
                    np.random.seed(42+j)
                    inds_to_shuffle = np.arange(len(target_y))
                    np.random.shuffle(inds_to_shuffle)
                    shuffled_y = [target_y[x] for x in inds_to_shuffle]
                    first_zero = shuffled_y.index(0)
                    if sum(shuffled_y[:3]) == 0:  # 3 rxns every iter
                        first_one = shuffled_y.index(1)
                        init_num_batches = first_one//3 + 1
                    else:
                        init_num_batches = first_zero//3 + 1
                    init_inds = inds_to_shuffle[:3*init_num_batches]
                    remaining_inds = [x for x in range(
                        len(target_y)) if x not in init_inds]
                    collected, remaining = divide_target_arrays(
                        target_desc, target_id, target_y,
                        init_inds, remaining_inds
                    )
                    
                    first_num_found = len(collected[2]) - sum(collected[2])
                    
                    rfc = RandomForestClassifier(max_depth=1, n_estimators=3*init_num_batches,
                                                random_state=42+j)
                    rfc.fit(collected[0], collected[2])
                    
                    rxns_collected_per_batch, _,\
                    model_by_iter, num_found_by_batch = explore_target_in_batches(
                                    rfc, collected[0], collected[2],
                                    remaining[0], remaining[1], remaining[2],
                                    "confidence", "add_collected", 
                                    weight_factor=1, num_trees_to_add=num_trees_to_add, new_max_depth=new_max_depth,
                                    num_rxns_per_batch=num_rxns_per_batch,
                                    print_progress=print_progress,
                                    random_state=42+j, enough_found=num_desired-first_num_found
                                )

                    rxns_collected_per_batch = [collected[num_rxns_per_batch*i:num_rxns_per_batch*(i+1)] for i in range(init_num_batches)] + rxns_collected_per_batch
                    model_by_iter = ["None"]*(init_num_batches-1) + [rfc] + model_by_iter
                    num_found_by_batch = [0]*(init_num_batches) + [first_num_found] + [x+first_num_found for x in num_found_by_batch[1:]]
                
                if len(num_found_by_batch) < (len_iter_result):
                    num_found_by_batch += [num_found_by_batch[-1]] * (len_iter_result-len(num_found_by_batch))
                    adding_tree_system[scenario]["num_rxns_found"] += num_found_by_batch
                elif len(num_found_by_batch) == len_iter_result:
                    adding_tree_system[scenario]["num_rxns_found"] += num_found_by_batch
                adding_tree_system[scenario]["num_rxns_conducted"] += [num_rxns_per_batch*x for x in range(len_iter_result-1)]
                adding_tree_system[scenario]["num_rxns_conducted"] += [len(target_y)]
                adding_tree_system[scenario]["Strategy"] += [strategy] * len_iter_result
                adding_tree_models[scenario]["models"] += [model_by_iter]
                adding_tree_models[scenario]["arrays"] += [rxns_collected_per_batch]

    return adding_tree_system, adding_tree_models


def plot_ATL_perf_comparison(
    source_list, adding_tree_system, 
    hue="Strategy", style="Strategy", ci=95,
    list_of_filenames=[], palette="plasma"
):
    """ Plots cumulative number of desired reactions found at each iteration
    for all strategies considered.
    
    Parameters
    ----------
    source_list : list of str
        Names of source nucleophiles.
    adding_tree_system : dict
        Output of function above.
    filename : list of str
        • if len(filename)==0 : does not save
        • else : save plots with given filenames. length should be same as source_list
    """
    for i, source in enumerate(source_list):
        scenario = f"{source}_to_heterocycle"
        fig, ax = plt.subplots()
        if palette == "plasma":
            sns.lineplot(x="num_rxns_conducted", y="num_rxns_found",
                    hue=hue, style=style, markers=True,
                    ci=ci, palette="plasma",
                    data=adding_tree_system[scenario])
        else : 
            sns.lineplot(x="num_rxns_conducted", y="num_rxns_found",
                         hue=hue, style=style, markers=True,
                         ci=ci, 
                         data=adding_tree_system[scenario])
        ax.set_xlabel("Number of Reactions Conducted", fontsize=14)
        ax.set_ylabel("Number of Reactions Found", fontsize=14)
        ax.set_yticks(2*np.arange(5))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        if len(list_of_filenames) > 0:
            fig.savefig(f"./figures/{list_of_filenames[i]}.pdf", format="pdf",
                            dpi=300, bbox_inches="tight")


