import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from update_combined_data import compare_weights

def passive_for_anil_or_alkyl(source_list, model_list, source_desc_list, source_y_list,
                              target_desc, target_id, target_y, target):
    """Conducts iterative labeling for aniline and alkyl amine target datasets.
    
    Parameters
    ----------
    source_list : list of str
        Source nucleophile names.
    model_list : list of classifiers
        List of source models.
    source_desc_list, source_y_list : list of np.2d/1darrays
        Source arrays of descriptors(input) and yield labels(output), respectively. 
    target_desc, target_id, target_y : np.2d/2d/1darrags
        Arrays of target descriptors(input), id(for easy interpretation) and yield labels(output), respectively.
    
    Returns
    -------
    perf_dict : dict
        • key : 
    """
    for i, source in enumerate(source_list):
        if i==0 :
            perf_dict = None
            model_dict = None
        perf_dict, model_dict = compare_weights(
                    source, model_list[i], [0],  # 0 because we won't be updating models.
                    source_desc_list[i], source_y_list[i], 
                    target_desc, target_id, target_y,
                    len(target_y)-sum(target_y), 
                    "new", perf_dict=perf_dict, model_dict=model_dict,
                    target=target
        )

    return perf_dict
            
def plot_passive_perfs(perf_dict, source_list, target, num_rxns_to_find,
                            save=False):

    for i,source in enumerate(source_list) :
        scenario = f"{source}_to_{target}"
        fig, ax = plt.subplots()
        sns.lineplot(x="num_rxns_conducted", y="num_rxns_found",
                     hue="weight", style="weight", markers=True,
                     alpha=0.7, ci=95, legend=False,
                     data=perf_dict[scenario], ax=ax)
        ax.set_xlabel("Number of Reactions Conducted", fontsize=14)
        ax.set_ylabel("Number of Reactions Found", fontsize=14)
        ax.set_yticks(np.arange(num_rxns_to_find+1))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        if save:
            fig.savefig(f"./figures/figureS5_{scenario}.pdf", format="pdf",
                        dpi=300, bbox_inches="tight")