import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AL import *

def compare_weights(source, model_list, weight_list, source_desc,
                    source_y, target_desc, target_id, target_y, enough_found,
                    model_update_strategy, add_every_iter=False, 
                    num_trees_to_add=0, new_max_depth=1,
                    perf_dict=None, model_dict=None, 
                    print_progress=False, target="heterocycle", random_state=42):
    ''' Repeats active learning experiments for all models,
    for all target data importance values (weights).
    
    Parameters
    ----------
    source : str
        name of nucleophile domain.
    model_list : list of RandomForestClassifiers.
        output of function above.
    weight_list : list of ints.
        importance values to consider.
        • zero corresponds to passive modeling (no update).
    source_desc, source_y : np.2d/1darrays
    target_desc, target_id, target_y : np.2d/2d/1darrays
    enough_found : int
        total number of reactions to be found.
    model_update_strategy : str
        • replace : replaces trees that predicts 'collected rxns' wrong.
        • add_all : adds new trees that are trained on combined data.
        --> should set 'num_trees_to_add' value to be greater than 0.
        • add_collected : adds new trees that are trained only on the newly collected data.
        --> should set 'num_trees_to_add' value to be greater than 0.
        • new : train new RF
        • none: use same model
    perf_dict, model_dict : None or dict
        if not None, append results to these dicts
    print_progress : bool
        whether to print which rxns were chosen and decision threshold values.
    random_state: int
    
    Returns
    -------
    all_active_perfs : dict
        stores results of all experiments.
    all_models : dict
        stores all models that were trained every iteration.
    '''
    if perf_dict is None and model_dict is None :
        all_active_perfs = {}
        all_models = {}
    else :
        all_active_perfs = copy.deepcopy(perf_dict)
        all_models = copy.deepcopy(model_dict)
    scenario = f"{source}_to_{target}"
    all_active_perfs.update({scenario:{
                "num_rxns_conducted":[],
                "num_rxns_found":[],
                "strategy":[],
                "weight":[]
        }})
    all_models.update({scenario:{
                "weight":[],
                "models":[],
                "arrays":[]
    }})    
    for i, model in enumerate(tqdm(model_list)) :
        #print(len(model.estimators_))
        for weight in weight_list :
            rxns_collected_per_batch, confidence_selected_rxns,\
            model_by_iter, num_found_by_batch = explore_target_in_batches(
                            model, source_desc, source_y,
                            target_desc, target_id, target_y,
                            "confidence", model_update_strategy, 
                            add_every_iter=add_every_iter,
                            num_trees_to_add=num_trees_to_add, 
                            new_max_depth=new_max_depth,
                            weight_factor=weight, 
                            enough_found=enough_found,
                            print_progress=print_progress,
                            random_state=random_state+i
                        )
            if len(num_found_by_batch)<16 : 
                num_found_by_batch += [num_found_by_batch[-1]]*(16-len(num_found_by_batch))
                all_active_perfs[scenario]["num_rxns_found"] += num_found_by_batch
            elif len(num_found_by_batch)==16:
                all_active_perfs[scenario]["num_rxns_found"] += num_found_by_batch
            all_active_perfs[scenario]["num_rxns_conducted"] += [3*x for x in range(15)]
            all_active_perfs[scenario]["num_rxns_conducted"] += [43]
            all_active_perfs[scenario]["strategy"]+=[model_update_strategy]*16
            all_active_perfs[scenario]["weight"] += [weight]*16
                
            all_models[scenario]["weight"] += [weight]
            all_models[scenario]["models"] += [model_by_iter]
            all_models[scenario]["arrays"] += [rxns_collected_per_batch]
    return all_active_perfs, all_models


def plot_AL_performance_by_weight(source, perf_dict, enough_found=8,
                                  ci=None, target="heterocycle",
                                  filename=None):
    ''' Plots average number of desired reactions found each
    iteration.
    
    Parameters
    ----------
    source : str
        nucleophile domain name.
    perf_dict : dict
        output of function above.
    enough_found : int
        number of desired rxns.
    ci : None or int
        confidence interval. 
    filename : str or None
    
    Returns
    -------
    None
    '''
    scenario = f"{source}_to_{target}"
    fig, ax = plt.subplots()
    sns.lineplot(x="num_rxns_conducted", y="num_rxns_found",
                 hue="weight", style="weight", markers=True,
                 alpha=0.7, ci=ci,
                 data=perf_dict[scenario])
    ax.set_xlabel("Number of Reactions Conducted", fontsize=14)
    ax.set_ylabel("Number of Reactions Found", fontsize=14)
    if enough_found == 8 :
        ax.set_yticks(2*np.arange(5))
    elif enough_found <=4 :
        ax.set_yticks(np.arange(enough_found+1))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    if filename is not None:
        fig.savefig(f"./figures/{filename}.pdf", format="pdf",
                    dpi=300, bbox_inches="tight")
