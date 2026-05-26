# ml_package 🧠

A **complete machine learning library built from scratch** using only NumPy.
API follows scikit-learn conventions — `fit()`, `transform()`, `predict()`.

## Install

```bash
pip install aishwarya_ml_package
```

## Package Structure

```
ml_package/
├── preprocessing/     StandardScaler, MinMaxScaler, Imputers, Outlier, Encoder
├── linear_models/     LinearRegression (OLS), Ridge, Lasso, ForwardSelection, BackwardElimination
├── decomposition/     PCA
├── neighbors/         KNNRegressor, KNNClassifier (Manhattan distance)
├── neural_network/    NeuralNetwork (backpropagation from scratch)
├── metrics/           r2, rmse, mae, mape | accuracy, precision, recall, f1
├── model_selection/   train_test_split, KFoldCV
├── pipeline/          Pipeline (chain steps like sklearn)
└── visualization/     PCA plots, scree chart, biplot
```

## Quick Start

```python
from ml_package.preprocessing   import StandardScaler, MedianImputer
from ml_package.linear_models   import LinearRegression
from ml_package.model_selection import train_test_split
from ml_package.pipeline        import Pipeline
from ml_package                 import metrics

# Full pipeline
pipe = Pipeline([
    ("imputer", MedianImputer()),
    ("scaler",  StandardScaler()),
    ("model",   LinearRegression()),
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
metrics.r2(y_test, preds)
```

## All Models

### LinearRegression — OLS Summary
```python
from ml_package.linear_models import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.ols_summary(feature_names=["Area", "BHK", "Price/sqft"])
model.predict(X_test)
```

### Ridge & Lasso
```python
from ml_package.linear_models import Ridge, Lasso
Ridge(alpha=1.0).fit(X, y).predict(X_test)
Lasso(alpha=0.01).fit(X, y).predict(X_test)
```

### Forward / Backward Feature Selection
```python
from ml_package.linear_models import ForwardSelection, BackwardElimination
fs = ForwardSelection(); fs.fit(X, y); fs.summary()
be = BackwardElimination(threshold_p=0.05); be.fit(X, y); be.summary()
```

### Check OLS Assumptions
```python
from ml_package.linear_models import check_assumptions
check_assumptions(X, y, model.predict(X), feature_names=["A","B","C"])
```

### KNN
```python
from ml_package.neighbors import KNNRegressor, KNNClassifier
KNNRegressor(k=5, distance="manhattan").fit(X, y).predict(X_test)
KNNClassifier(k=5, weights="distance").fit(X, y).predict(X_test)
```

### Neural Network
```python
from ml_package.neural_network import NeuralNetwork
nn = NeuralNetwork(hidden_layers=(16, 8), learning_rate=0.01, epochs=2000)
nn.fit(X_train, y_train)
nn.predict(X_test)
nn.summary()
nn.plot_loss()
```

### PCA
```python
from ml_package.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
pca.summary()
pca.plot_variance()
```

### Pipeline
```python
from ml_package.pipeline import Pipeline
pipe = Pipeline([
    ("imputer", MedianImputer()),
    ("scaler",  StandardScaler()),
    ("model",   NeuralNetwork(hidden_layers=(16, 8)))
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
pipe.summary()
```

### Cross-Validation
```python
from ml_package.model_selection import KFoldCV
from ml_package.metrics         import r2
kf = KFoldCV(n_splits=5, random_state=42)
kf.cross_val_score(LinearRegression(), X, y, metric_fn=r2)
```

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

## Upload to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

## License
MIT
