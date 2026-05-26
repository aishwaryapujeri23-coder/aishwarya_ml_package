"""
ml_package
==========
A complete machine learning library built from scratch using only NumPy.
API mirrors scikit-learn conventions (fit/transform/predict).

Modules
-------
preprocessing   : StandardScaler, MinMaxScaler, MeanImputer, MedianImputer,
                  ModeImputer, IQROutlierRemover, ZScoreOutlierRemover,
                  LabelEncoder, OneHotEncoder, L1Normalizer, L2Normalizer
linear_models   : LinearRegression (OLS), Ridge, Lasso,
                  ForwardSelection, BackwardElimination,
                  check_assumptions, OLSSummary
decomposition   : PCA
neighbors       : KNNRegressor, KNNClassifier
neural_network  : NeuralNetwork
metrics         : r2, rmse, mae, mape (regression)
                  accuracy, precision, recall, f1,
                  confusion_matrix, classification_report
model_selection : train_test_split, KFoldCV
pipeline        : Pipeline
visualization   : plot_pca_2d, plot_scree, plot_biplot

Quick Start
-----------
>>> from ml_package.linear_models   import LinearRegression
>>> from ml_package.preprocessing   import StandardScaler, MedianImputer
>>> from ml_package.model_selection import train_test_split
>>> from ml_package.metrics         import r2, rmse
>>> from ml_package.pipeline        import Pipeline
>>>
>>> pipe = Pipeline([
...     ("imputer", MedianImputer()),
...     ("scaler",  StandardScaler()),
...     ("model",   LinearRegression()),
... ])
>>> pipe.fit(X_train, y_train)
>>> print(r2(y_test, pipe.predict(X_test)))
"""

from aishwarya_ml_package.preprocessing   import (StandardScaler, MinMaxScaler,
                                         MeanImputer, MedianImputer, ModeImputer,
                                         IQROutlierRemover, ZScoreOutlierRemover,
                                         LabelEncoder, OneHotEncoder,
                                         L1Normalizer, L2Normalizer)
from aishwarya_ml_package.linear_models   import (LinearRegression, Ridge, Lasso,
                                         ForwardSelection, BackwardElimination,
                                         check_assumptions, OLSSummary)
from aishwarya_ml_package.decomposition   import PCA
from aishwarya_ml_package.neighbors       import KNNRegressor, KNNClassifier
from aishwarya_ml_package.neural_network  import NeuralNetwork
from aishwarya_ml_package.model_selection import train_test_split, KFoldCV
from aishwarya_ml_package.pipeline        import Pipeline
from aishwarya_ml_package                 import metrics

__version__ = "1.0.0"
__author__  = "ml_package"
