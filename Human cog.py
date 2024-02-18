#!/usr/bin/env python
# coding: utf-8

# In[39]:


#import all libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[40]:


#import the dataset
df = pd.read_csv('phase_transfer_entropy_exp_grp.csv')


# In[41]:


train_df = df
df


# In[42]:


#train_df = train_df.drop(18, axis=0) #Removed rows 18, 19, 38, 39 for holdout testing
#train_df


# In[43]:


data = train_df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)


# In[44]:


from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV

# define model
#model = Ridge()
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
#model = SVR()
model = XGBRegressor()
#model = MLPRegressor()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()

##Ridge Regression
#space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
#space['alpha'] = loguniform(1e-5, 100)
#space['fit_intercept'] = [True, False]

##Decision Tree
#space['max_depth']= [2, 3, 4, 5, 6, 7, 8, 9, 10]
#space['min_samples_split']= [2, 3, 4, 5, 6, 7, 8, 9, 10]
#space['min_samples_leaf']= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

##Random Forest
#space['max_depth']= [2, 3, 4, 5, 6, 7, 8, 9, 10]
#space['min_samples_split']= [2, 3, 4, 5, 6, 7, 8, 9, 10]
#space['min_samples_leaf']= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#space['n_estimators']= [10, 20, 30, 35, 40, 50, 100]

##SVR
#space['C'] = [0.01, 0.1, 1, 10, 100]
#space['gamma'] = [0.001, 0.01, 0.1, 1]
#space['kernel'] = ['linear', 'rbf', 'poly']

##XGBoost
space['learning_rate'] = [0.01, 0.1, 0.5]
space['max_depth'] = [3, 5, 7, 10]
space['subsample'] = [0.6, 0.8, 1.0]
space['colsample_bytree'] = [0.6, 0.8, 1.0]
space['n_estimators'] = [10, 20, 30, 35, 50, 100, 150]

##MLP
#space['hidden_layer_sizes'] = [(50,), (100,)]
#space['activation'] = ['logistic', 'tanh', 'relu']
#space['solver'] = ['adam', 'lbfgs', 'sgd']
#space['alpha'] = [0.01, 0.1]

# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# In[45]:


model = XGBRegressor(subsample = 1.0, n_estimators = 100, max_depth = 5, learning_rate = 0.1, colsample_bytree = 0.8)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
errors = mean_squared_error(y_test, predictions, squared=False)
print(errors)


# In[46]:


import shap
from sklearn.model_selection import GridSearchCV


# In[47]:


# Create a SHAP explainer and generate SHAP values for the test data
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_test)


# In[ ]:


from xgboost import plot_importance
from matplotlib import pyplot
# plot feature importance
plot_importance(model)
pyplot.show()


# In[48]:


from xgboost import plot_importance
from matplotlib import pyplot
# plot feature importance
plot_importance(model, max_num_features=10)
pyplot.show()


# In[14]:


import matplotlib.pyplot as plt
import xgboost as xgb



# Define the mapping of original feature names to replacements
feature_replacements = {
    'f0': 'somatosensory area (left hemisphere) x somatosensory area (right hemisphere)',
    'f2': 'somatosensory area (left hemisphere) x posterior somatosensory area (right hemisphere)',
    'f5': 'somatosensory area (left hemisphere) x primary somatosensory cortex [cytoarchitectonic area 2] (left hemisphere)',
    'f404': 'primary motor area (posterior, left hemisphere) x perirhinal cortex (right hemisphere)',
    'f158': 'primary somatosensory cortex [cytoarchitectonic area 1] (right hemisphere) x entorhinal cortex (left hemisphere)',
    'f6': 'somatosensory area (left hemisphere) x primary somatosensory cortex [cytoarchitectonic area 2] (right hemisphere)',
    'f407': 'primary motor area (posterior, right hemisphere) x posterior somatosensory area (right hemisphere)',
    'f202': 'primary somatosensory cortex [cytoarchitectonic area 2] (right hemisphere) x primary motor area (posterior, left hemisphere)',
    'f49': 'somatosensory area (right hemisphere) x secondary visual area (right hemisphere)',
    'f4': 'somatosensory area (left hemisphere) x primary somatosensory cortex [cytoarchitectonic area 1] (right hemisphere)',
}

# Update the y-axis labels of the feature importance graph
fig, ax = plt.subplots(figsize=(10, 12))
xgb.plot_importance(model, ax=ax, height=0.8,max_num_features=10)
ax.set_yticklabels([feature_replacements.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()])
plt.show()


# In[ ]:




