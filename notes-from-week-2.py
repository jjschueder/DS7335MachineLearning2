# notes from week 2

# inputs
# list / dict of classifiers and hyperparamters

# for each classifier

    # build hyparameter grid (sklearn)

    # for each parameter set

        # set the parameters of the model

        # change params with .set_params(**params)

        # we can put this in a function
        # k-fold cross valiation
        # store the scores (dict) (mean scores)
        # return dictionary


# build scoring function

def k_fold_score(model,
                 metrics: list,
                 X_data: np.array,
                 y_data: np.array,
                 splits: int):
    """
    model: classification model; already constructed
    metrics: list of callables with signature metric(y_true, y_pred)
    X_data: X training data
    y_data: y training data
    splits: number of k-fold splits
    """
    
    # for fold in k-folds
    #   fit on fold
    #   predict
    #       for metric in metrics
    #           Hint: use metric.__name__ as keys to build dict of scores
    #           score on metric
    #           store performance
    
    # for metric in metrics
    #   average values
    
    # return dict{metric: average_value}


# something like this returned from k_fold_score:
#{
#    'accuracy_score': [2]
#    'another_metric': [15]
#}



# hyperparameter grid

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html


# k-fold validation

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html




# setting hyperparameters

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.set_params

# positional vs keyword args

arg1 and arg2 are "positional args"
arg3 is "keyword argument"

# lets assume we have a function like this
def function(arg1, arg2, arg3=True)

args = [2,3] # list for positional args
kwargs = {   # dict for keyword args
    'arg1': 1
    'arg2': 23
    'arg3': False
}

# we can 'unpack' the args into a function call
function(*args, **kwargs)

# which corresponds to this
function(arg1, arg2, arg3=False)


# example - why is this useful

from sklearn.neighbors import KNeighborsClassifier

# this constructs KNeighborsClassifier with the default parameters
knn_model = KNeighborsClassifier()

# see what parameters are used as default
knn_model.get_params()

# how can we change them
# set some parameters
knn_parameters = {
    'n_neighbors': 3,
    'algorithm': 'ball_tree'
}
# change them
knn_model.set_params(**knn_parameters)

# same as knn_model.set_params(n_neighbors=3, algorithm='ball_tree')

# see that they've changed
knn_model.get_params()



# scorers / metrics

# metrics from sklearn from names
from sklearn.metrics import accuracy_score

# special method that gets object name
accuracy_score.__name__

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html