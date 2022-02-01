import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from AL import *

DESC_IDX = [0, 6, 33, 58, 65, 73]
COMPONENT_LIST = ["Nucleophile", "Electrophile", "Catalyst", "Base", "Solvent"]


##############################################################################
#############           Analyzing selected descriptors           #############
##############################################################################
def determine_component(feature_num, desc_idx=DESC_IDX):
    ''' Determines which component the descriptor comes from.
    
    Parameters
    ----------
    feature_num : int
        descriptor index
    desc_idx : list of ints
        index at which component descriptor starts.
    
    Returns
    -------
    i : int
        component index.
    '''
    for i in range(5):
        if feature_num >= desc_idx[i] and feature_num < desc_idx[i+1]:
            return i


def get_component_portion_per_batch(source, model_dict, num_models=25,
                                    num_weights_considered=3, weight_idx=1):
    '''Out of  ALL models considered for the AL experiment,
    computes which component descriptors are used in trees trained
    after each iteration.
    
    Parameters
    ----------
    source : str
        domain name.
    model_dict : dict
        result of AL experiment.
    num_weights_considered : int
    weight_idx : int
        index that we are interested in.
        
    Returns
    -------
    {source,batch1,batch2,batch3}_portion, : list
        portion of each components descriptor used in trees
        trained in each batch.
    '''
    source_count = [0, 0, 0, 0, 0]
    batch1_count = [0, 0, 0, 0, 0]
    batch2_count = [0, 0, 0, 0, 0]
    batch3_count = [0, 0, 0, 0, 0]

    for i in range(num_models):
        for j, dtc in enumerate(model_dict[f"{source}_to_heterocycle"]["models"][num_weights_considered*i+weight_idx][4].estimators_):
            feature = dtc.tree_.feature[0]
            if feature != -2:
                comp = determine_component(feature)
                if j < 5:
                    source_count[comp] += 1
                elif j < 8:
                    batch1_count[comp] += 1
                elif j < 11:
                    batch2_count[comp] += 1
                elif j < 14:
                    batch3_count[comp] += 1

    source_portion = [x/sum(source_count) for x in source_count]
    batch1_portion = [x/sum(batch1_count) for x in batch1_count]
    batch2_portion = [x/sum(batch2_count) for x in batch2_count]
    batch3_portion = [x/sum(batch3_count) for x in batch3_count]

    return source_portion, batch1_portion, batch2_portion, batch3_portion


def plot_component_portions(source_portion, batch1_portion, batch2_portion, batch3_portion,
                            component_list=COMPONENT_LIST, filename=None):
    ''' Plots the portions of descriptor components used in
    each batch of trees. '''
    labels = ['Source', 'Iter 1', 'Iter 2', 'Iter 3']

    x = np.arange(5)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*3/2, source_portion, width,
                    label='Source', color="#440154") #
    rects2 = ax.bar(x - width/2, batch1_portion, width,
                    label='Iter 1', color='#bad6eb') #"#39568C"
    rects3 = ax.bar(x + width/2, batch2_portion, width,
                    label="Iter 2", color='#89bedc') #"#1F968B"
    rects4 = ax.bar(x + width*3/2, batch3_portion, width,
                    label="Iter 3", color='#539ecd') #"#95D840"

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_yticks([0.2*x for x in range(6)])
    ax.set_yticklabels([round(0.2*x, 1) for x in range(6)])
    ax.set_ylabel('Proportion of Descriptors', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(component_list)
    ax.set_xlabel("Reaction Component", fontsize=14)
    for i in range(4):
        if i != 0:
            ax.axvline(i+0.5, 0, 1, c="gray", ls="--")
        else:
            ax.axvline(i+0.5, 0, 0.65, c="gray", ls="--")
    ax.legend(loc="upper left")
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    if filename is not None:
        fig.savefig(f"./figures/{filename}.pdf",
                    format="pdf", dpi=300, bbox_inches="tight")


##############################################################################
#############            Analyzing selected reactions            #############
##############################################################################
def plot_conducted_rxns_by_pca(list_of_selected_rxns_per_batch,
                               target_id, target_desc,
                               xmin=-4, xmax=7, ymin=-5, ymax=8,
                               color_list=["#39568C", "#1F968B",
                                           "#95D840", "#FDE725", "tab:orange"],
                               marker_dict={0: "*", 1: "x"},
                               filename=None):
    ''' For a single model instance, plots on a PCA plot of which reactions
    were selected every iteration up to the fifth batch.
    Desired reactions are marked with * and opposite with x.'''
    ### PCA Transformation
    scaler = StandardScaler()
    # not necessary to transform nuc, elec
    target_std = scaler.fit_transform(target_desc[:, 33:])
    pca = PCA(n_components=40)
    target_pca = pca.fit_transform(target_std)[:, :2]
    ### Data Jittering to avoid overlap of points
    for i, id_row in enumerate(target_id):
        if id_row[3] == 2:
            target_pca[i, 1] += 0.323
        #elif id_row[3]==1:
        #    target_pca[i,1]+= 0.55
        elif id_row[3] == 4:
            target_pca[i, 1] -= 0.4
        elif id_row[3] == 3:
            target_pca[i, 1] += 0.1
    eigen_vals = pca.singular_values_
    ### Plot of all target rxns
    fig, ax = plt.subplots()
    ax.set_aspect(eigen_vals[1]/eigen_vals[0])
    sel_idx = []
    ### Collecting conducted rxns each batch
    for i, (rxns_desc, rxns_id, rxns_y) in enumerate(list_of_selected_rxns_per_batch):
        if i < len(color_list):
            for j, rxn in enumerate(rxns_desc):
                ind = np.where(np.all(rxn == target_desc, axis=1))[0][0]
                sel_idx.append(ind)
                marker = marker_dict[rxns_y[j]]
                ax.scatter(x=target_pca[ind, 0], y=target_pca[ind, 1],
                           c=color_list[i], marker=marker, alpha=1, s=60)
    unsel_idx = [x for x in range(target_pca.shape[0]) if x not in sel_idx]
    ax.scatter(x=target_pca[unsel_idx, 0],
               y=target_pca[unsel_idx, 1], c='grey', s=30, alpha=0.4)

    # Adding Marker Legend First
    marker_legend_elements = [
        Line2D([0], [0], markerfacecolor="grey", marker='o',
               color="w", label="Unlabeled", markersize=10, alpha=0.4),
    ]
    if marker_dict[0] == "*" :
        marker_legend_elements += [Line2D([0], [0], markerfacecolor=color_list[0], marker=marker_dict[0], color="w", label="Positive", markersize=10)]
    elif marker_dict[0] == "+":
        marker_legend_elements += [Line2D([0], [0], markeredgecolor=color_list[0], marker=marker_dict[0],
                color="none", label="Positive", markersize=10)]
    if marker_dict[1] == "_" :
        marker_legend_elements += [Line2D([0], [0], color=color_list[0], linewidth=1.5, label="Negative")]
    else :
        marker_legend_elements += [Line2D([0], [0], markeredgecolor=color_list[0], marker=marker_dict[1],
               color="none", label="Negative", markersize=10)]
    marker_legend = ax.legend(handles=marker_legend_elements,
                              bbox_to_anchor=(1, 1), title="Reaction Label")
    plt.gca().add_artist(marker_legend)
    # Then add Batch Legend
    order_list = ["1st", "2nd", "3rd", "4th", "5th"]
    legend_elements = [
        Line2D([0], [0], color=color_list[x], label=order_list[x], markersize=30)
        for x in range(len(color_list))
    ]
    ax.legend(handles=legend_elements,
              bbox_to_anchor=(1.01, 1.0), title="Batch")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_xticklabels([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel("PC1 (Catalyst)", fontsize=14)
    ax.set_ylabel("PC2 (Solvent, Base)", fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    if filename is not None:
        fig.savefig(f"./figures/figure8D_{filename}.pdf",
                    format="pdf", dpi=300, bbox_inches="tight")


def plot_conducted_rxns_by_comp_ids(list_of_selected_rxns_per_batch,
                                    color_list=["#39568C", "#1F968B",
                                                "#95D840", "#FDE725", "tab:orange"],
                                    marker_dict={0: "*", 1: "x"},
                                    filename=None, legend_inside=False,
                                    show_source=False):
    ''' On a grid-table divided by component id, reactions chosen each batch is marked.'''
    cat_coord_dict = {1: 1, 2: 2, 8: 3, 10: 4, 11: 5, 15: 6}

    fig, ax = plt.subplots()
    for i, (rxns_desc, rxns_id, rxns_y) in enumerate(list_of_selected_rxns_per_batch):
        if i < len(color_list):
            for j, rxn in enumerate(rxns_id):
                marker = marker_dict[rxns_y[j]]
                ax.scatter(x=cat_coord_dict[rxn[2]]+(rxn[-1]-1.5)*0.5,
                           y=rxn[3], s=100,
                           c=color_list[i],
                           marker=marker)
    if show_source:
        ax.scatter(x=[0.75, 1.75], y=[4, 4], marker="*",
                   c="grey", s=100, alpha=0.5)
    ax.set_yticks([x+1 for x in range(4)])
    ax.set_ylim(0.5, 4.5)
    ax.set_ylabel("Base ID", fontsize=14)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticklabels(["", "1", "2", "8", "10", "11", "15"])
    ax.set_xlabel("Catalyst ID", fontsize=14)
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Solvent ID', fontsize=14)
    secax.set_xticks([0.5*x+0.75 for x in range(12)])
    secax.set_xticklabels(["1", "2"]*6)
    for i in range(3):
        if legend_inside:
            if i < 2:
                ax.axhline(i+1.5, 0, 1, c="grey")
            else:
                ax.axhline(i+1.5, 0, 4.9/6, c="grey")
        else:
            ax.axhline(i+1.5, 0, 1, c="grey")
    for j in range(6):
        if legend_inside:
            if j != 5:
                ax.axvline(j+1, 0, 1, c="grey", ls="--", alpha=0.5)
            else:
                ax.axvline(j+1, 0, 0.535, c="grey", ls="--", alpha=0.5)
            if j != 4:
                ax.axvline(j+1.5, 0, 1, c="grey")
            else:
                ax.axvline(5.5, 0, 0.535, c="grey")
        else:
            ax.axvline(j+1, 0, 1, c="grey", ls="--", alpha=0.5)
            ax.axvline(j+1.5, 0, 1, c="grey")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    order_list = ["1st", "2nd", "3rd", "4th", "5th"]
    if show_source:
        legend_elements = [
            Line2D([0], [0], markerfacecolor="grey", marker='*',
                   color="w", label="Source", markersize=15)
        ]
    else:
        legend_elements = []
    legend_elements += [
        Line2D([0], [0], markerfacecolor=color_list[x], marker='*',
               color="w", label=order_list[x], markersize=15)
        for x in range(len(color_list))
    ]
    if legend_inside:
        ax.legend(handles=legend_elements, loc="upper right", title="Batch",
                  fontsize=8)
    else:
        ax.legend(handles=legend_elements,
                  bbox_to_anchor=(0.99, 1), title="Batch")
    if filename is not None:
        fig.savefig(f"./figures/figure9_{filename}.pdf",
                    format="pdf", dpi=300, bbox_inches="tight")


##############################################################################
#############            Analyzing selected reactions            #############
##############################################################################
def compare_num_target_trees(source_list, source_model_list, source_desc_list, source_y_list,
                             target_desc, target_id, target_y,):
    adding_diff_tree_system = {}

    for i, source in enumerate(source_list):
        # Preparing Random Models
        model_list = source_model_list[i]
        scenario = f"{source}_to_heterocycle"
        adding_diff_tree_system.update({scenario: {
            "num_rxns_conducted": [],
            "num_rxns_found": [],
            "Number of\nTrees Added": []
        }})

        for j, model in enumerate(tqdm(model_list)):
            for num_trees_to_add in [1, 3, 5]:
                model_to_use = copy.deepcopy(model)
                print(f"        Starting with {len(model.estimators_)} Trees")
                rxns_collected_per_batch, confidence_selected_rxns,\
                    model_by_iter, num_found_by_batch = explore_target_in_batches(
                        model_to_use, source_desc_list[i], source_y_list[i],
                        target_desc, target_id, target_y,
                        "confidence", "add_collected",
                        num_trees_to_add=num_trees_to_add,
                        print_progress=False,
                        random_state=42+j
                    )
                if len(num_found_by_batch) < 16:
                    num_found_by_batch += [num_found_by_batch[-1]
                                        ]*(16-len(num_found_by_batch))
                    adding_diff_tree_system[scenario]["num_rxns_found"] += num_found_by_batch
                elif len(num_found_by_batch) == 16:
                    adding_diff_tree_system[scenario]["num_rxns_found"] += num_found_by_batch
                adding_diff_tree_system[scenario]["num_rxns_conducted"] += [
                    3*x for x in range(15)]
                adding_diff_tree_system[scenario]["num_rxns_conducted"] += [43]
                adding_diff_tree_system[scenario]["Number of\nTrees Added"] += [
                    num_trees_to_add]*16
    return adding_diff_tree_system


##############################################################################
#############               Analyzing adaptability               #############
##############################################################################
def get_roc_and_probs_by_batch(dict_of_models, source, model_ind,
                                   weight_ind, source_desc, source_y,
                                   num_weights_considered=4, num_batches=16):
    ''' Prepares arrays of how models updated at the end of each iteration makes predictions
    on data in hand (source data + target data collected upto that batch).
    
    Parameters
    ----------
    dict_of_models : dict
        all_models dict four cells above this.
    source : {"amides", "sulfonamides", "ROH"}
        source domain.
    model_ind, weight_ind : int
        • index of model = 25*(model_num)+(ind_of_weight)
        • index of weight
    source_desc, source_y : np.2d/1darrays
        Arrays of source descriptors(input) and yield labels(output).
    num_batches : int
        number of batches = num_target_rxns//num_rxns_per_batch + 2 (initial and end)
    
    Returns
    -------
    roc_source_rxns, roc_collected_rxns : np.1darray
        updated model's ROC on source rxns only / collected target rxns only
    prob_desired_rxns : np.1darray
        Average predicted probability values of target rxns with desired outcomes.
    '''
    scenario = f"{source}_to_heterocycle"
    models_by_batch = dict_of_models[scenario]["models"][num_weights_considered *
                                                         model_ind+weight_ind]
    arrays_by_batch = dict_of_models[scenario]["arrays"][num_weights_considered *
                                                         model_ind+weight_ind]
    roc_source_rxns = np.zeros(num_batches)
    roc_collected_rxns = np.zeros(num_batches)
    prob_desired_rxns = np.zeros(num_batches)

    for i, model in enumerate(models_by_batch):
        if i == 0:
            (X, y) = arrays_by_batch[0][0], arrays_by_batch[0][2]
        else:
            arrays_to_consider = arrays_by_batch[:i]
            X = np.vstack(tuple([x[0] for x in arrays_to_consider]))
            y = np.concatenate(tuple([x[2] for x in arrays_to_consider]))
        # ROC-AUC of model on source reactions
        roc = roc_auc_score(source_y,
                            model.predict_proba(source_desc)[:, 1])
        roc_source_rxns[i+1] = roc
        # ROC-AUC of model on collected target rxns up to this batch
        proba = model.predict_proba(X)[:, 1]
        if sum(y) not in [len(y), 0]:
            roc_collected_rxns[i+1] = roc_auc_score(y, proba)
            prob_desired_rxns[i+1] = np.mean(proba[np.where(y == 0)[0]])
        else:
            roc_collected_rxns[i+1] = 0
            if sum(y) == len(y):
                prob_desired_rxns[i+1] = 0
            else:
                prob_desired_rxns[i+1] = np.mean(proba[np.where(y == 0)[0]])
        # Predicted probabilities of collected 'desired rxns'

    if len(models_by_batch) < num_batches-2:  # -2 due to the first and last
        roc_collected_rxns[i+2:] = roc_collected_rxns[i+1]
        prob_desired_rxns[i+2:] = prob_desired_rxns[i+1]
        roc_source_rxns[i+2:] = roc_source_rxns[i+1]

    return roc_source_rxns, roc_collected_rxns, prob_desired_rxns


def prep_dict_to_plot_AL_and_ROC(AL_result_dict, source, model_ind,
                                 weight_ind, roc_source_rxns, roc_collected_rxns, num_enough=8,
                                 num_weights_considered=4, num_batches=16,
                                 dict_to_plot=None):
    ''' Prepares a dictionary for plotting ROC-AUC values of each model along with
    active learning performance.
    
    Parameters
    ----------
    AL_result_dict : dict
        corresponds to all_active_perfs 4 cells above
    source : {"amides", "sulfonamides", "ROH"}
        source domain.
    roc_source_rxns, roc_collected_rxns : np.1darray
        outcomes of function above
    
    Returns
    -------
    dict_to_plot : dict
    '''
    start_ind = model_ind*weight_ind*num_batches + weight_ind*num_batches
    if dict_to_plot is None:
        dict_to_plot = {
            "num_rxns_conducted": [],
            "score": [],
            "measure": [],
        }
    scenario = f"{source}_to_heterocycle"

    for i in range(2):
        dict_to_plot["num_rxns_conducted"] += AL_result_dict[scenario]['num_rxns_conducted'][start_ind:start_ind+num_batches]

    roc_names = ["Source ROC", "Target ROC"]
    for i in range(2):
        dict_to_plot["measure"] += [roc_names[i]]*num_batches

    dict_to_plot["score"] += list(roc_source_rxns)
    dict_to_plot["score"] += list(roc_collected_rxns)

    return dict_to_plot


def prep_dict_to_plot_AL_and_prob(AL_result_dict, source, model_ind,
                                  weight_ind, prob_desired_rxns,
                                  num_batches=16, dict_to_plot=None):
    ''' Prepares a dictionary for plotting predicted probability values of desired target rxns by each model.
    
    Parameters
    ----------
    AL_result_dict : dict
        corresponds to all_active_perfs 4 cells above
    source : {"amides", "sulfonamides", "ROH"}
        source domain.
    model_ind : int
        index of the model of interest.
    weight_ind : int
        index of the weight value of interest within the list of weight values evaluated.
    prob_desired_rxns : np.1darray
        outcomes of function above
    num_batches : int
        maximum number of iterations.
    dict_to_plot : dict
        if we want to concatenate to previous results.
    
    Returns
    -------
    dict_to_plot : dict
    '''
    start_ind = model_ind*weight_ind*num_batches + weight_ind*num_batches
    if dict_to_plot is None:
        dict_to_plot = {
            "num_rxns_conducted": [],
            "score": [],
            "measure": [],
        }
    scenario = f"{source}_to_heterocycle"

    dict_to_plot["num_rxns_conducted"] += AL_result_dict[scenario]['num_rxns_conducted'][start_ind:start_ind+num_batches]
    dict_to_plot["measure"] += ["Predicted Target Prob"]*num_batches
    dict_to_plot["score"] += [0]
    dict_to_plot["score"] += list(1-prob_desired_rxns[1:])

    return dict_to_plot


def plot_AL_and_ROC(dict_to_plot, ylabel_second_half,
                    filename=None, ci=None):
    """ Plots either the ROC-AUC or predicted probability on source/collected target data
    using models updated after each iteration to evaluate how well the ATL strategy
    adapts in the target reaction space.

    Parameters
    ----------
    dict_to_plot : dict
        output of function above.
    ylabel_second_half : str
        y-axis label
    filename : str or None
        if str : saves the plot with filename.
        if None : does not save.
    ci : None or int
        confidence interval %.
    """
    fig, ax = plt.subplots()
    if ylabel_second_half != "Avg. Pred. Proba.":
        sns.lineplot(x="num_rxns_conducted", y="score",
                     hue="measure", style="measure", markers=True,
                     data=dict_to_plot, hue_order=["Source ROC", "Target ROC"])
    else:
        sns.lineplot(x="num_rxns_conducted", y="score",
                     hue="measure", style="measure", markers=True,
                     data=dict_to_plot, palette=["tab:green"],
                     ci=ci)
    ax.set_xlabel("Number of Reactions Conducted", fontsize=14)
    ax.set_ylabel(f"{ylabel_second_half}", fontsize=14)
    ax.set_yticks([round(0.2*x, 1) for x in range(6)])
    ax.set_yticklabels([round(0.2*x, 1) for x in range(6)])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    if filename is not None:
        fig.savefig(f"./figures/{filename}.pdf",
                    format="pdf", dpi=300, bbox_inches="tight")


##############################################################################
#############          Analyzing number of target trees          #############
##############################################################################
def update_perf_dict(dict_to_update, num_rxns_found_by_batch, num_max_iter, num_rxns_per_batch, target_y,
):
    dict_to_update["num_rxns_found"] += num_rxns_found_by_batch
    dict_to_update["num_rxns_conducted"] += [num_rxns_per_batch*x for x in range(num_max_iter-1)]
    dict_to_update["num_rxns_conducted"] += [len(target_y)]


def compare_num_trees(source_list, source_model_list, source_desc_list, source_y_list,
                      target_desc, target_id, target_y,
                      list_num_trees_to_add=[1,3,5], new_max_depth=1, num_rxns_per_batch=3, 
                      print_progress=False):
    adding_diff_tree_system = {}
    num_desired = len(target_y) - sum(target_y)

    for i, source in enumerate(tqdm(source_list)):
        # Preparing Random Models
        model_list = source_model_list[i]
        scenario = f"{source}_to_heterocycle"
        adding_diff_tree_system.update({scenario: {
            "num_rxns_conducted": [],
            "num_rxns_found": [],
            "Number of\nTrees Added": []
        }})

        for j, model in enumerate(model_list):
            for num_trees_to_add in list_num_trees_to_add:
                model_to_use = copy.deepcopy(model)
                print(f"        Starting with {len(model.estimators_)} Trees")
                _, _, _, num_found_by_batch = explore_target_in_batches(
                        model_to_use, source_desc_list[i], source_y_list[i],
                        target_desc, target_id, target_y,
                        "confidence", "add_collected",
                        num_trees_to_add=num_trees_to_add,
                        new_max_depth=new_max_depth,
                        enough_found=num_desired,
                        print_progress=print_progress,
                        random_state=42+j
                    )
                len_iter_result = len(target_y)//num_rxns_per_batch + 2
                if len(num_found_by_batch) < len_iter_result:
                    num_found_by_batch += [num_found_by_batch[-1]]*(len_iter_result-len(num_found_by_batch))
                update_perf_dict(
                    adding_diff_tree_system[scenario], num_found_by_batch,
                    len_iter_result, num_rxns_per_batch, target_y
                )
                adding_diff_tree_system[scenario]["Number of\nTrees Added"] += [num_trees_to_add]*len_iter_result
    return adding_diff_tree_system


##############################################################################
#############       Analyzing reaction selection strategies      #############
##############################################################################
def compare_strategies(source_list, source_model_list, source_desc_list, source_y_list,
                       target_desc, target_id, target_y, list_of_std_coeffs=[0.5,2], num_rxns_per_batch=3,
                       num_trees_to_add=3, print_progress=False,
                       ):
    adding_tree_ucb_system = {}
    len_iter_result = len(target_y)//num_rxns_per_batch + 2

    for i, source in enumerate(tqdm(source_list)):
        # Preparing Random Models
        model_list = source_model_list[i]
        scenario = f"{source}_to_heterocycle"
        adding_tree_ucb_system.update({scenario: {
            "num_rxns_conducted": [],
            "num_rxns_found": [],
            "Strategy": [],
            "std coefficient": []
        }})

        for k, model in enumerate(model_list):
            model_to_use = copy.deepcopy(model)
            for strategy in ["exploitation", "ucb", "exploration"]:
                if strategy == "ucb":
                    for coeff in list_of_std_coeffs:
                        _, _, _, num_found_by_batch = explore_target_in_batches(
                                model_to_use, source_desc_list[i], source_y_list[i],
                                target_desc, target_id, target_y,
                                "ucb", "add_collected",
                                num_trees_to_add=num_trees_to_add,
                                coeff=coeff,
                                print_progress=print_progress,
                                random_state=42+k
                            )
                        adding_tree_ucb_system[scenario]["Strategy"] += [strategy]*len_iter_result
                        adding_tree_ucb_system[scenario]["std coefficient"] += [coeff]*len_iter_result
                        if len(num_found_by_batch) < len_iter_result:
                            num_found_by_batch += [num_found_by_batch[-1]
                                                ] * (len_iter_result-len(num_found_by_batch))
                        update_perf_dict(
                            adding_tree_ucb_system[scenario], num_found_by_batch,
                            len_iter_result, num_rxns_per_batch, target_y
                        )

                elif strategy == "exploration":
                    _, _, _, num_found_by_batch = explore_target_in_batches(
                            model_to_use, source_desc_list[i], source_y_list[i],
                            target_desc, target_id, target_y,
                            "variance", "add_collected",
                            num_trees_to_add=num_trees_to_add,
                            print_progress=print_progress,
                            random_state=42+k
                        )
                    adding_tree_ucb_system[scenario]["Strategy"] += [strategy]*len_iter_result
                    adding_tree_ucb_system[scenario]["std coefficient"] += [0]*len_iter_result
                    if len(num_found_by_batch) < len_iter_result:
                        num_found_by_batch += [num_found_by_batch[-1]
                                               ] * (len_iter_result-len(num_found_by_batch))
                    update_perf_dict(
                        adding_tree_ucb_system[scenario], num_found_by_batch,
                        len_iter_result, num_rxns_per_batch, target_y
                    )

                else:
                    _, _, _, num_found_by_batch = explore_target_in_batches(
                            model_to_use, source_desc_list[i], source_y_list[i],
                            target_desc, target_id, target_y,
                            "confidence", "add_collected",
                            num_trees_to_add=num_trees_to_add,
                            print_progress=print_progress,
                            random_state=42+k
                        )
                    adding_tree_ucb_system[scenario]["Strategy"] += [strategy]*len_iter_result
                    adding_tree_ucb_system[scenario]["std coefficient"] += [0]*len_iter_result

                    if len(num_found_by_batch) < len_iter_result:
                        num_found_by_batch += [num_found_by_batch[-1]] * (len_iter_result-len(num_found_by_batch))
                    update_perf_dict(
                        adding_tree_ucb_system[scenario], num_found_by_batch,
                        len_iter_result, num_rxns_per_batch, target_y
                    )
    return adding_tree_ucb_system

##############################################################################
#############     Analyzing impact of source model complexity    #############
##############################################################################
def compare_source_model_hyperparam(
    source_list, source_model_list, source_desc_list, source_y_list,
    target_desc, target_id, target_y, hyperparam, hyperparam_val_list, 
    num_rxns_per_batch=3, num_trees_to_add=3, print_progress=False,
):
    perf_by_hyperparam = {}
    len_iter_result = len(target_y)//num_rxns_per_batch + 2
    
    for i, source in enumerate(tqdm(source_list)):
        # Preparing Random Models
        model_list = source_model_list[i]
        scenario = f"{source}_to_heterocycle"
        perf_by_hyperparam.update({scenario: {
            "num_rxns_conducted": [],
            "num_rxns_found": [],
            "Source Model": []
        }})
        all_source_models = []
        if hyperparam == "depth":
            for depth in hyperparam_val_list :
                all_source_models.append(prepare_models(
                    source_desc_list[i], source_y_list[i], 25, n_estimators=5, max_depth=depth
                ))
        elif hyperparam == "num_trees":
            for n_trees in hyperparam_val_list:
                all_source_models.append(prepare_models(
                    source_desc_list[i], source_y_list[i], 25, n_estimators=n_trees, max_depth=1
                ))
        else :
            print("Invalid hyperparameter to investigate.")
            break

        for j in range(25):
            for k in range(len(hyperparam_val_list)):
                _, _, _, num_found_by_batch = explore_target_in_batches(
                    all_source_models[k][j], source_desc_list[i], source_y_list[i],
                        target_desc, target_id, target_y,
                        "confidence", "add_collected",
                        num_trees_to_add=num_trees_to_add,
                        print_progress=print_progress,
                        random_state=42+j
                    )
                if len(num_found_by_batch) < len_iter_result:
                    num_found_by_batch += [num_found_by_batch[-1]] * \
                        (len_iter_result-len(num_found_by_batch))
                update_perf_dict(perf_by_hyperparam[scenario], num_found_by_batch, len_iter_result, num_rxns_per_batch, target_y)
                if hyperparam == "depth":
                    perf_by_hyperparam[scenario]["Source Model"] += [f"Depth {hyperparam_val_list[k]}"]*len_iter_result
                else :
                    perf_by_hyperparam[scenario]["Source Model"] += [f"{hyperparam_val_list[k]} Trees"]*len_iter_result
    return perf_by_hyperparam


