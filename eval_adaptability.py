import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

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
