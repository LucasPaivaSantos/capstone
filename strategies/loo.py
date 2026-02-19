from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from .base import BaseStrategy, register_strategy

@register_strategy('leave-one-out')
class LeaveOneOutStrategy(BaseStrategy):
    def __init__(self, seed):
        super().__init__(seed)
        print(f"Leave-One-Out Cross-Validation Strategy initialized")

    def evaluate(self, model, X, y, **kwargs):
        n_samples = len(X) # one sample per input entry
        print(f"Running Leave-One-Out Cross-Validation ({n_samples} iterations)")

        loo = LeaveOneOut()

        y_true_all = []
        y_pred_all = []
        
        iteration = 1
        for train_index, test_index in loo.split(X):
            print(f"  Processing iteration {iteration}/{n_samples}")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # create a fresh model instance for each iteration
            from models.base import MODEL_REGISTRY
            model_cls = type(model)
            fold_model = model_cls(seed=model.seed)
            
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            # add the values of this iteration to the overall
            y_true_all.append(y_test.values[0])
            y_pred_all.append(y_pred[0])
            
            iteration += 1
        
        # calculate overall metrics
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        r2 = r2_score(y_true_all, y_pred_all)
        
        return {
            "RMSE": rmse,
            "R2": r2,
            "n_iterations": n_samples
        }