import numpy as np
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

######################################################
#####               Utility  Funcs               #####
######################################################
def prepare_models(desc_array, y_array, num_models,
                   n_estimators=5, max_depth=1, random_state=42):
    ''' Prepares differently initiated models.
    
    Parameters
    ----------
    desc_array, y_array : np.2d/1darrays
    num_models : int
        number of models to prepare.
    other arguments : hyperparameters of random forest classifier.
    
    Returns
    -------
    model_list : list of RandomForestClassifiers.
    '''
    model_list = []
    for i in range(num_models):
        rfc = RandomForestClassifier(random_state=random_state+i,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)
        rfc.fit(desc_array, y_array)
        model_list.append(rfc)
    return model_list


def concat_source_and_batch(source_X, source_y,
                            target_batch_X, target_batch_y,
                            sample_weight_array,
                            weight_factor):
    ''' Concatenates input arrays, prepares sample_weight_array
    required for .fit(). All outputs will be used as input for
    next round.
    
    Parameters
    ----------
    source_X, source_y : np.2d/1darrays
        Source arrays
    target_batch_X, target_batch_y : np.2d/1darrays
        Arrays of target reactions that were collected this iteration.
    sample_weight_array : list of ints
        Source:1, target:some integer
    weight_factor: int
        Sample weight for newly collected target data.
    
    Outputs
    -------
    new_X, new_y : np.2d/1darray
        Concatenated X, y arrays
    sample_weight_array : list
        Concatenated weight_array
    '''
    new_X = np.vstack(tuple([source_X]+[target_batch_X]))
    new_y = np.concatenate(tuple([source_y]+[target_batch_y]))
    sample_weight_array += [weight_factor]*len(target_batch_y)
    return new_X, new_y, sample_weight_array


def divide_target_arrays(X_desc_target, X_id_target, y_target,
                         idx_rxn_to_run, remaining_idx):
    ''' Splits the target array after simulating experiment conduction.
    
    Parameters
    ----------
    X_desc/id/y_target : np.2darray
        Arrays to split
    idx_rxn_to_run : list
        Indices of reactions in which exp has been conducted
    remaining_idx : list
        Indices of reactions that has not yet been experimented.
    
    Returns
    -------
    rxns_collected, rxns_remaining : tuple of np.ndarrays
    '''
    #print("inds collected: ", idx_rxn_to_run)
    desc_collected = X_desc_target[idx_rxn_to_run,:]
    id_collected = X_id_target[idx_rxn_to_run,:]
    y_collected = y_target[idx_rxn_to_run]
    
    desc_remaining = X_desc_target[remaining_idx,:]
    id_remaining = X_id_target[remaining_idx,:]
    y_remaining = y_target[remaining_idx]
    
    assert desc_collected.shape[0]+desc_remaining.shape[0]==X_desc_target.shape[0]
    rxns_collected = (desc_collected, id_collected, y_collected)
    rxns_remaining = (desc_remaining, id_remaining, y_remaining)
    return rxns_collected, rxns_remaining


######################################################
#####               Choosing  Rxns               #####
######################################################

def most_confident_rxns(model, X_target_desc, num_rxns):
    ''' Chooses the reactions to conduct based on highest confidence.
    
    Parameters
    ----------
    model : classifier
        Current model.
    X_target_desc : np.2darray
        Remaining desc/id target reaction candidates
    num_rxns : int
        Number of reactions to select.
        
    Returns
    -------
    idx_rxn_to_run : np.1darray
        Indices of rxns to run within X_target_desc
    remaining_idx : np.1darray
        Indices of rxns remaining within X_target_desc
    confidence : np.1darray
        Predicted probability values of selected reactions.
    '''
    full_pred_proba = model.predict_proba(X_target_desc)
    if full_pred_proba.shape[1] > 1:
        pred_proba = full_pred_proba[:,1]
    else : 
        print("Only one label predicted", full_pred_proba)
        pred_proba = full_pred_proba
    confidence = 1-np.sort(pred_proba)[:num_rxns]
    idx_rxn_to_run = np.argsort(pred_proba)[:num_rxns]
    remaining_idx = np.argsort(pred_proba)[num_rxns:]
    return idx_rxn_to_run, remaining_idx, confidence


def highest_variance_btw_trees(model, X_target_desc, num_rxns):
    ''' Chooses reactions which have highest variance of predicted
    probability values across all trees in the RF model.
    
    Parameters
    ----------
    model : classifier
        Current model.
    X_target_desc : np.2darray
        Target descriptor array.
    num_rxns : int
        Number of reactions to select.
        
    Returns
    -------
    idx_rxn_to_run : np.1darray
        Indices of rxns to run within X_target_desc
    remaining_idx : np.1darray
        Indices of rxns remaining within X_target_desc
    confidence : np.1darray
        Predicted probability values of selected reactions.
    '''
    all_proba = np.zeros((X_target_desc.shape[0],
                          len(model.estimators_)))
    for i,tree in enumerate(model.estimators_):
        all_proba[:,i] = tree.predict_proba(X_target_desc)[:,0]
    var_proba = np.var(all_proba, axis=1)
    pred_proba = model.predict_proba(X_target_desc)[:,1]
    idx_rxn_to_run = np.argsort(var_proba)[-num_rxns:]
    remaining_idx = np.argsort(var_proba)[:-num_rxns]
    return idx_rxn_to_run, remaining_idx, 1-pred_proba[idx_rxn_to_run]


def highest_expectation_plus_std(model, X_target_desc, num_rxns,
                                 coeff):
    ''' Follows upper confidence bound. Selects reactions with
    highest predicted probability values + std across trees.
    
    Parameters
    ----------
    model : classifier
    X_target_desc : np.2darray
        Target descriptor array.
    num_rxns : int
        Number of reactions to select.
    coeff : float
        For upper confidence bound, coefficient of standard deviation.
        (Coefficient of average is 1.)
        
    Returns
    -------
    idx_rxn_to_run : np.1darray
        Indices of rxns to run within X_target_desc
    remaining_idx : np.1darray
        Indices of rxns remaining within X_target_desc
    pred_proba : np.1darray
        Predicted probability values of selected reactions.
    '''
    pred_proba = model.predict_proba(X_target_desc)[:,0]
    all_proba = np.zeros((X_target_desc.shape[0],
                          len(model.estimators_)))
    for i,tree in enumerate(model.estimators_):
        all_proba[:,i] = tree.predict_proba(X_target_desc)[:,0]
    std_proba = np.std(all_proba, axis=1)
    pred_proba = model.predict_proba(X_target_desc)[:,0]
    comb = pred_proba + coeff*std_proba
    idx_rxn_to_run = np.argsort(comb)[-num_rxns:]
    remaining_idx = np.argsort(comb)[:-num_rxns]
    return idx_rxn_to_run, remaining_idx, pred_proba

######################################################
#####               Updating Model               #####
######################################################

def replace_trees(model, new_X, new_y,
                  newly_conducted_X, newly_conducted_y,
                  sample_weight_array,
                  max_depth, random_state):
    '''Within the random forest, replaces trees that predicted 
    at least one result of the conducted experiments incorrectly.
    
    Parameters
    ----------
    new_X, new_y : np.ndarrays
        Arrays of the combined source+collected target data.
    newly_conducted_X/y : np.ndarrays
        Arrays of the collected target data only.
    sample_weight_array : list
        Sample weights for training new model
    max_depth : int
        Maximum depth of new tree
    random_state : int
    
    Returns
    -------
    model : random forest classifier
    '''
    tree_inds_to_replace = []
    for i,tree in enumerate(model.estimators_):
        pred_by_tree = tree.predict(newly_conducted_X)
        if len(np.where(pred_by_tree!=newly_conducted_y)[0])!=0:
            tree_inds_to_replace.append(i)
    print(f"    Replacing {len(tree_inds_to_replace)} Trees.")
    if len(tree_inds_to_replace) > 0:
        if len(tree_inds_to_replace)==1:
            dtc = DecisionTreeClassifier(random_state=random_state,
                                         max_features="sqrt",
                                         max_depth=max_depth)
            dtc.fit(new_X, new_y, sample_weight=sample_weight_array)
            model.estimators_[tree_inds_to_replace[0]]=dtc
        else : 
            rfc = RandomForestClassifier(
                            random_state=random_state,
                            max_depth=max_depth,
                            n_estimators = len(tree_inds_to_replace)
                    )
            rfc.fit(new_X, new_y, sample_weight=sample_weight_array)
            for i, ind in enumerate(tree_inds_to_replace) :
                model.estimators_[ind] = rfc.estimators_[i]
    return model


def add_trees(model, new_X, new_y, sample_weight_array,
              num_trees, max_depth, random_state):
    '''Adds specified number of trees while not changing
    the original model.
    
    Parameters
    ----------
    model : classifier
        Current model.
    new_X, new_y : np.ndarrays
        Arrays of the reactions to train new trees on.
    sample_weight_array : list
        Sample weights for training new model
    num_trees : int
        Number of trees to add to the forest.
    max_depth : int
        Maximum depth of new tree
    random_state : int
    
    Returns
    -------
    model : random forest classifier
    '''
    rfc = RandomForestClassifier(
                        random_state=random_state,
                        max_depth=max_depth,
                        n_estimators=num_trees,
                )
    rfc.fit(new_X, new_y, sample_weight=sample_weight_array)
    copied_model = copy.deepcopy(model)
    copied_model.estimators_ += rfc.estimators_
    copied_model.n_estimators+= rfc.n_estimators

    return copied_model

######################################################
#####      Actual Iterative Active Learning      #####
######################################################
def explore_target_in_batches(
    source_model, X_desc_source, y_source,
    X_desc_target, X_id_target, y_target,
    rxn_selection_strategy, 
    model_update_strategy, add_every_iter=True,
    num_trees_to_add=0, new_max_depth=1,
    num_rxns_per_batch=3, weight_factor=3, 
    enough_found=8, random_state=42, coeff=None,
    print_progress=True
):
    ''' Simulates exploration by conducting experiments selected
    by model in batches.
    
    Parameters
    ----------
    source_model : classifier
        Model to be initially used.
    X_desc_source, y_source : np.2d/1darrays
        Descriptor (input) and yield label (output) arrays of source data.
    X_desc_target, X_id_target, y_target : np.2d/2d/1darrays
        Descriptor (input), ID (to easily track reactions) and 
        yield label (output) arrays of target data.
    rxn_selection_strategy : {"confidence", "confusion", "variance"}
                             or a list with these elements
        Strategy to sample reactions to conduct.
    model_update_strategy : {"replace", "add_all", "add_collected", "new"}
        • replace : replaces trees that predicts 'collected rxns' wrong.
        • add_all : adds new trees that are trained on combined data.
        --> should set 'num_trees_to_add' value to be greater than 0.
        • add_collected : adds new trees that are trained only on the newly collected data.
        --> should set 'num_trees_to_add' value to be greater than 0.
        • new : train new RF
        • none: use same model
    add_every_iter : bool
        whether new trees are added every iteration.
        • True : number of trees will be [5,8,11,14,•••] if num_trees_to_add=3
        • False : number of trees will be [5,8,8,8,8,•••] if num_trees_to_add=3
    num_trees_to_add : int
        if model_update_strategy=="add" : 
            number of trees to add each iteration.
        if model_update_strategy=="new" : 
            new model will be trained with previous model's 
            number of trees + num_trees_to_add
    new_max_depth : int or list
        • int : fixes the max_depth value across iterations
        • list : changes max_depth value as iteration progresses.
    num_rxns_per_batch : int
        number of reactions to sample each batch.
    weight_factor : int
        weight each collected target data will have,
        compared to source data when being combined
    enough_found : int
        after finding this number of reactions, stop exploring
    random_state : int
        random seed for modeling etc.
    coeff : float
        if upper confidence bound is considered, use as coefficient of standard deviation.
    print_progress : Bool
        whether to print information on sampled reactions each batch.
        
    Returns
    -------
    rxns_collected_per_batch : list of tuples
        (X_id_array, X_desc_array, y_array) per each iteration.
    confidence_selected_rxns : list of np.1darray
        confidence on selected reactions' desiredness for every iteration.
    model_by_iter : list of classifiers
        models used to select the reactions to conduct each iter.
    num_found_by_batch : list of ints
        cumulative number of desired reactions found after every iteration.
    '''
    ### 0) Initiation
    rxns_collected_per_batch = [] 
    confidence_selected_rxns = []
    model_by_iter = []
    num_found_by_batch = [0]
    total_num_found = 0
    sample_weight_array = [1]*len(y_source)
    num_trees = len(source_model.estimators_)
    iteration = 1
    ref_source_model = copy.deepcopy(source_model)
    ref_source_desc = copy.deepcopy(X_desc_source)
    while total_num_found < enough_found :
        if print_progress:
            print("-----------"+"-"*len(str(iteration)))
            print(f"Iteration {iteration}")
            print("-----------"+"-"*len(str(iteration)))
        ### 1) Reaction Selection Phase
        if type(rxn_selection_strategy) == str:
            rxn_selection = rxn_selection_strategy
        elif type(rxn_selection_strategy) == list:
            rxn_selection = rxn_selection_strategy[iteration-1]
            
        if rxn_selection=="confidence":
            idx_rxn_to_run, remaining_idx, conf = most_confident_rxns(
                            source_model, X_desc_target, num_rxns_per_batch
                        )
        elif rxn_selection=="variance":
            idx_rxn_to_run, remaining_idx, conf = highest_variance_btw_trees(
                            source_model, X_desc_target, num_rxns_per_batch
                        )
        elif rxn_selection=="ucb":
            idx_rxn_to_run, remaining_idx, conf = highest_expectation_plus_std(
                            source_model, X_desc_target, num_rxns_per_batch, coeff)
        else :
            print("Inappropriate reaction selection strategy.")
            break

        ### 1-1) Update items to return
        #print("Before dividing", idx_rxn_to_run)
        rxns_collected, rxns_remaining = divide_target_arrays(
                            X_desc_target, X_id_target, y_target,
                            idx_rxn_to_run, remaining_idx
                        )
        rxns_collected_per_batch.append(rxns_collected)
        confidence_selected_rxns.append(conf)
        num_found = len(np.where(y_target[idx_rxn_to_run]==0)[0]) # 0 because negative is minor for our dataset
        total_num_found += num_found
        num_found_by_batch.append(total_num_found)
    
        if print_progress:
            print(np.hstack((rxns_collected[1][:,2:], 
                             rxns_collected[2].reshape(-1,1))))
            print(f"Found {num_found} rxns.")
            print()
            
        ### 1-2) Update arrays and sample_weight
        if weight_factor != 0:
            #print("Collected shape:", rxns_collected[0].shape)
            X_desc_source, y_source, sample_weight_array = concat_source_and_batch(
                                       X_desc_source, y_source,
                                       rxns_collected[0], rxns_collected[2],
                                       sample_weight_array, weight_factor)
        (X_desc_target, X_id_target, y_target) = rxns_remaining
        
        ### 2) Update Model
        if weight_factor!=0:
            if type(new_max_depth) == int:
                max_depth = new_max_depth
            elif type(new_max_depth) == list:
                max_depth = new_max_depth[iteration-1]

            if model_update_strategy=="replace":
                source_model = replace_trees(
                        source_model, X_desc_source, y_source, 
                        rxns_collected[0], rxns_collected[2],
                        sample_weight_array,
                        max_depth, random_state+iteration
                )
            elif model_update_strategy=="add_all":
                if add_every_iter:
                    source_model = add_trees(
                        source_model, X_desc_source, y_source, 
                        sample_weight_array, num_trees_to_add, 
                        max_depth, random_state+iteration
                    )
                else:
                    source_model = add_trees(
                        ref_source_model, X_desc_source, y_source, 
                        sample_weight_array, num_trees_to_add, 
                        max_depth, random_state+iteration
                    )
                #print("Number of trees: ", source_model.n_estimators)
            elif model_update_strategy=="add_collected":
                if add_every_iter:
                    source_model = add_trees(
                            source_model, rxns_collected[0], rxns_collected[2], 
                            [1]*len(rxns_collected[2]), num_trees_to_add, 
                            max_depth, random_state+iteration
                        )
                else : 
                    #print("Number of RXNS to train on: ", len(y_source[ref_source_desc.shape[0]:]))
                    source_model = add_trees(
                            ref_source_model, 
                            X_desc_source[ref_source_desc.shape[0]:,:], 
                            y_source[ref_source_desc.shape[0]:], 
                            [1]*len(y_source[ref_source_desc.shape[0]:]), num_trees_to_add, 
                            max_depth, random_state+iteration
                        )
                #print("Number of trees: ", source_model.n_estimators)
            elif model_update_strategy=="new":
                rfc = RandomForestClassifier(random_state=random_state+iteration,
                                             max_depth=max_depth,
                                             n_estimators=num_trees+num_trees_to_add)
                rfc.fit(X_desc_source, y_source, 
                        sample_weight=sample_weight_array)
                source_model = rfc
            elif model_update_strategy=="none":
                source_model = source_model
            else : 
                print("Inappropriate model update strategy.")
                break
        
        model_by_iter.append(copy.deepcopy(source_model))
        iteration+= 1
        #print(f"Source: {X_desc_source.shape[0]}, Target: {X_desc_target.shape[0]}")
    return rxns_collected_per_batch, confidence_selected_rxns,\
           model_by_iter, num_found_by_batch
