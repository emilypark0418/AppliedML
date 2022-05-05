#!/usr/bin/env python
# coding: utf-8

# # Homework 2: Trees and Calibration
# 
# 
# ## Instructions
# 
# Please push the .ipynb, .py, and .pdf to Github Classroom prior to the deadline. Please include your UNI as well.
# 
# **Make sure to use the dataset that we provide in CourseWorks/Classroom. DO NOT download it from the link provided (It may be different).**
# 
# Due Date : 03/02 (2nd March), 11:59 PM EST

# ## Name: Chaewon Park
# 
# ## UNI: cp3227

# ## The Dataset
# Credit ([Link](https://www.kaggle.com/gamersclub/brazilian-csgo-plataform-dataset-by-gamers-club?select=tb_lobby_stats_player.csv) | [License](https://creativecommons.org/licenses/by-nc-sa/4.0/))
# 
# The goal is to predict wins based on in match performace of multiple players. Please use this dataset and this task for all parts of the assignment.
# 
# ### Features
# 
# idLobbyGame - Categorical (The Lobby ID for the game)
# 
# idPlayer - Categorical (The ID of the player)
# 
# idRooom - Categorical (The ID of the room)
# 
# qtKill - Numerical (Number of kills)
# 
# qtAssist - Numerical (Number of Assists)
# 
# qtDeath - Numerical (Number of Deaths)
# 
# qtHs - Numerical (Number of kills by head shot)
# 
# qtBombeDefuse - Numerical (Number of Bombs Defuses)
# 
# qtBombePlant - Numerical (Number of Bomb plants)
# 
# qtTk - Numerical (Number of Team kills)
# 
# qtTkAssist - Numerical Number of team kills assists)
# 
# qt1Kill - Numerical (Number of rounds with one kill)
# 
# qt2Kill - Numerical (Number of rounds with two kill)
# 
# qt3Kill - Numerical (Number of rounds with three kill)
# 
# qt4Kill - Numerical (Number of rounds with four kill)
# 
# qt5Kill - Numerical (Number of rounds with five kill)
# 
# qtPlusKill - Numerical (Number of rounds with more than one kill)
# 
# qtFirstKill - Numerical (Number of rounds with first kill)
# 
# vlDamage - Numerical (Total match Damage)
# 
# qtHits - Numerical (Total match hits)
# 
# qtShots - Numerical (Total match shots)
# 
# qtLastAlive - Numerical (Number of rounds being last alive)
# 
# qtClutchWon - Numerical (Number of total clutchs wons)
# 
# qtRoundsPlayed - Numerical (Number of total Rounds Played)
# 
# descMapName - Categorical (Map Name - de_mirage, de_inferno, de_dust2, de_vertigo, de_overpass, de_nuke, de_train, de_ancient)
# 
# vlLevel - Numerical (GC Level)
# 
# qtSurvived - Numerical (Number of rounds survived)
# 
# qtTrade - Numerical (Number of trade kills)
# 
# qtFlashAssist - Numerical (Number of flashbang assists)
# 
# qtHitHeadshot - Numerical (Number of times the player hit headshot
# 
# qtHitChest - Numerical (Number of times the player hit chest)
# 
# qtHitStomach - Numerical (Number of times the player hit stomach)
# 
# qtHitLeftAtm - Numerical (Number of times the player hit left arm)
# 
# qtHitRightArm - Numerical (Number of times the player hit right arm)
# 
# qtHitLeftLeg - Numerical (Number of times the player hit left leg)
# 
# qtHitRightLeg - Numerical (Number of times the player hit right leg)
# 
# flWinner - Winner Flag (**Target Variable**).
# 
# dtCreatedAt - Date at which this current row was added. (Date)
# 

# ## Question 1: Decision Trees

# **1.1: Load the provided dataset**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


# In[2]:


## unzip dataset
#import zipfile
#with zipfile.ZipFile("tb_lobby_stats_player.csv.zip", 'r') as zip_ref:
#    zip_ref.extractall("./")


# In[3]:


player_df = pd.read_csv('tb_lobby_stats_player.csv')

player_df.head()


# **1.2: Plot % of missing values in each column. Would you consider dropping any columns? Assuming we want to train a decision tree, would you consider imputing the missing values? If not, why? (Remove the columns that you consider dropping - you must remove the dtCreatedAt column)**

# No I would not drop the columns. Dropping the columns only make sense if a lot of the values in the individual column is missing. However, as plotted below, the number of missing values in the columns are less than 1%. Hence, it would be better to impute or infer the missing values rather than removing the columns.
# 
# For training a decision tree, I used the mean values of the columns to fill (impute) in the missing values. This is because sklearn's implementation does not support missing values.

# In[5]:


col = player_df.columns.values[3 :]
refined_player_df = player_df[col]

sum_df = refined_player_df.isnull().sum()

print(sum_df)


# In[7]:


data = (sum_df != 0)
null_cols = data.index[data]

plt.bar(null_cols, refined_player_df.isnull().sum()[null_cols] * 100 / refined_player_df.isnull().count()[null_cols])
plt.xlabel("Columns")
plt.ylabel("% of Missing Values")
plt.rcParams["figure.figsize"] = (27, 5)

plt.show()


# In[8]:


player_df = player_df.drop(['dtCreatedAt'], axis = 1)


# In[9]:


mean_value=player_df[null_cols].mean()
print(mean_value)

player_df_new = player_df.copy(deep=True)

# Replace NaNs with mean of values of the same column
player_df_new = player_df_new.fillna(value=mean_value)


# In[10]:


player_df = player_df_new
player_df.isnull().sum()


# **1.3: Plot side-by-siide bars of class distribtuion for each category for the categorical feature and the target categories.**

# In[11]:


key = player_df['descMapName'].value_counts().keys()
width =1 
for i in range(len(key)):
    k = key[i]
    plt.bar(i , player_df['descMapName'].value_counts()[k], width, label = str(k))

plt.xticks(np.arange(len(key)), key)
plt.xlabel("descMapName")
plt.ylabel("Count Distribution")
plt.show()
plt.rcParams["figure.figsize"] = (20, 5)


# In[13]:


key = player_df['flWinner'].value_counts().keys()

for i in range(len(key)):
    k = key[i]
    plt.bar(i , player_df['flWinner'].value_counts()[k], width, label = str(k))

plt.xticks(np.arange(len(key)), key)
plt.xlabel("flWinner")
plt.ylabel("Count Distribution")
plt.show()
plt.rcParams["figure.figsize"] = (5, 4)


# **1.4: Split the data into development and test datasets. Which splitting methodology did you choose and why?**

# I chose to randomly split into dev and test set because we have a decent amount of balanced data, so k-folds is unnecessary.

# In[14]:


player_df_X = player_df.drop(columns=['flWinner'])
player_df_y = player_df['flWinner']


# In[15]:


x_dev, x_test, y_dev, y_test = train_test_split(player_df_X, player_df_y, test_size=0.2, random_state= 418)


# **1.5: Preprocess the data (Handle the Categorical Variable). Do we need to apply scaling? Briefly Justify**

# I used target encoding to preprocess the categorial variables because using one-hot encoding and then scaling could cause problems. I also applied scaling on the numerical data so that they are normalized with a mean of 0 and var of 1. Currently their range are from 0 (or 1) to various values and is not centered at 0.

# In[16]:


### Your code here (Numerical data range checking)
num_features= player_df_X.drop(columns=['idLobbyGame', 'idPlayer', 'idRoom', 'descMapName']).columns.values
te_features = ['idLobbyGame', 'idPlayer', 'idRoom', 'descMapName']

player_df_X[num_features].mean()


# In[17]:


player_df_X[num_features].min()


# In[18]:


player_df_X[num_features].max()


# In[19]:


from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from category_encoders import TargetEncoder
from sklearn import pipeline


# In[20]:


# use target encoder
te = TargetEncoder(cols=te_features).fit(x_dev, y_dev)
x_dev = te.transform(x_dev)
x_test = te.transform(x_test)

# use standard scaler()
ss = StandardScaler()
x_dev = ss.fit_transform(x_dev)
x_test = ss.transform(x_test)


# In[25]:


print(x_dev.shape, x_test.shape)


# Note: Both x_dev and x_test have the same number of categories as well as same categories due to our splitting method, so we don't need to apply encoding twice. 

# **1.6: Fit a Decision Tree on the development data until all leaves are pure. What is the performance of the tree on the development set and test set? Provide metrics you believe are relevant and briefly justify.**

# The basic score works well in this case, because the dataset is balanced so we don't have to worry about the model making biased decisions that may incur the need to use F1 score.

# In[21]:


DT = DecisionTreeClassifier()
pipe = pipeline.make_pipeline(DT)
pipe.fit(x_dev, y_dev)

print(pipe.score(x_dev, y_dev))
print(pipe.score(x_test, y_test))


# **1.7: Visualize the trained tree until the max_depth 8**

# In[22]:


fig = plt.figure(figsize=(25,20))
_ = plot_tree(DT, max_depth=8, feature_names= player_df_X.columns, class_names= ['0', '1'], filled=True)


# **1.8: Prune the tree using one of the techniques discussed in class and evaluate the performance**

# In[26]:


# As alpha increases, more of the tree is pruned, which increases the total impurity of its leaves
# the maximum effective alpha value is removed, because it is the trivial tree with only one node
path = DT.cost_complexity_pruning_path(x_test, y_test)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")


# In[27]:


print(ccp_alphas)
print(ccp_alphas[-11:-1])


# Since, there are approximately 9.5k alpha values for the pruning path, you can just use last 10 values of alpha excluding the last value.

# In[28]:


pipe = pipeline.make_pipeline(GridSearchCV(DT, param_grid = [{"ccp_alpha": ccp_alphas[-11:-1]}], return_train_score = True))
pipe.fit(x_dev, y_dev)

grid_results = pipe.named_steps["gridsearchcv"]
print("Best Score:", grid_results.best_score_)
print("Best Alpha:", grid_results.best_params_)
print("Best Score:", pipe.score(x_test, y_test))


# As seen above, pruning the tree increased the test score from 0.7271863375960468 to 0.7609622329016318.

# **1.9: List the top 3 most important features for this trained tree? How would you justify these features being the most important?**

# The top 3 most important features are: qtSurvived, qtDeath, idRoom.
# 
# The importance of these features can be justified by the fact that they are located on the top of the decision tree. This means that they have the highest information gain and therefore the most important features with highest discriminative power. Also, if we take the indexes of the Decision Tree's top 3 feature importances, it corresponds to the features listed above.

# In[29]:


DT.ccp_alpha = 0.0013093455812577676
DT.fit(x_dev, y_dev)

feat_imp = DT.feature_importances_

col_list = player_df_X.columns.values

k=3
index = np.argpartition(feat_imp, len(feat_imp) - k)[-k:]
print("Top 3 Most Important Features:", col_list[index])


# In[30]:


fig = plt.figure(figsize=(25,20))
_ = plot_tree(DT, max_depth=2, feature_names= player_df_X.columns, class_names= ['0', '1'], filled=True)


# ## Question 2: Random Forests

# In[31]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
pipe = pipeline.make_pipeline(RFC)
pipe.fit(x_dev, y_dev)

print(pipe.score(x_dev, y_dev))
print(pipe.score(x_test, y_test))


# **2.1: Train a Random Forest model on the development dataset using RandomForestClassifier class in sklearn. Use the default parameters. Evaluate the performance of the model on test dataset. Does this perform better than Decision Tree on the test dataset (compare to results in Q 1.6)?**

# In Q.16, we had dev score of 1.0 and test score of 0.7271863375960468.
# For this question, we similarly have dev score of 1.0 but a much higher score of 0.7895251282886698. Hence Random Forest model works better than Decision Tree model on the test dataset.

# **2.2: Does all trees in the trained random forest model have pure leaves? How would you verify this?**

# Yes, we can verify this by looking at the gini impurity value of all the leaves in each individual tree (estimator).

# In[32]:


ImpureFlag = False
count=0
for tree in RFC.estimators_:
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    leaf_nodes = []
    for i in range(n_nodes):
        if children_left[i] == children_right[i]:
            if tree.tree_.impurity[i]!=0:
                ImpureFlag = True
                print("Impure Leaf Node Exists")
                break
    count= count+1
if not ImpureFlag:
    print("All", count , " Trees have Pure Leaves")


# **2.3: Assume you want to improve the performance of this model. Also, assume that you had to pick two hyperparameters that you could tune to improve its performance. Which hyperparameters would you choose and why?**
# 

# I would choose the number of trees (n_estimators) and the number of features (max_features). This is because they are the most intuitive and most significant hyper-parameters that we can gauge the effects of.

# **2.4: Now, assume you had to choose up to 5 different values (each) for these two hyperparameters. How would you choose these values that could potentially give you a performance lift?**

# I would randomly choose the values so that we can do random search. Random search has empirically proven to be more effective than grid search.

# In[33]:


from sklearn.model_selection import RandomizedSearchCV
import random

model_params = {
    'n_estimators': random.sample(range(1, 180), 3),
    'max_features': random.sample(range(1, RFC.n_features_), 3)
}


# **2.5: Perform model selection using the chosen values for the hyperparameters. Use cross-validation for finding the optimal hyperparameters. Report on the optimal hyperparameters. Estimate the performance of the optimal model (model trained with optimal hyperparameters) on test dataset? Has the performance improved over your plain-vanilla random forest model trained in Q2.1?**

# As seen below, the model score improved from 0.7895251282886698 to 0.793244820938883.

# In[34]:


pipe = pipeline.make_pipeline(RandomizedSearchCV(RFC, model_params, return_train_score = True, cv = 5, verbose = 5))
pipe.fit(x_dev, y_dev)

random_results = pipe.named_steps["randomizedsearchcv"]
print("Best Score:", random_results.best_score_)
print("Best Params:", random_results.best_params_)
print("Best Score:", pipe.score(x_test, y_test))


# **2.6: Can you find the top 3 most important features from the model trained in Q2.5? How do these features compare to the important features that you found from Q1.9? If they differ, which feature set makes more sense?**

# The top 3 most important features are: 'vlDamage' 'qtDeath' 'qtSurvived'.
# This is slightly different from the previous answer of 'qtSurvived', 'qtDeath', and 'idRoom'.
# Using 'vlDamage' seems more plausible than 'idRoom', because 'vlDamage' is a numerical variable that refers to 'total match damage' and 'idRoom' is simply a categorical feature representing the ID of the room.
# The target variable, "winner flag" has a stronger relationship with the match damage, compared to the room ID.

# In[35]:


RFC.n_estimators = 109
RFC.max_features = 19

RFC.fit(x_dev, y_dev)

feat_imp = RFC.feature_importances_

col_list = player_df_X.columns.values

k=3
index = np.argpartition(feat_imp, len(feat_imp) - k)[-k:]
print("Top 3 Most Important Features:", col_list[index])


# ## Question 3: Gradient Boosted Trees

# **3.1: Choose three hyperparameters to tune GradientBoostingClassifier and HistGradientBoostingClassifier on the development dataset using 5-fold cross validation. Report on the time taken to do model selection for both the models. Also, report the performance of the test dataset from the optimal models.**

# In[38]:


import time
from sklearn.ensemble import GradientBoostingClassifier

start_time = time.time()

GBC = GradientBoostingClassifier()

model_params = {
    'n_estimators': random.sample(range(1, 200), 3),
    'max_features': random.sample(range(1, DT.n_features_), 3),
    'learning_rate': [0.001, 0.01, 0.1]
}

pipe = pipeline.make_pipeline(RandomizedSearchCV(GBC, model_params, return_train_score = True, cv = 5, verbose = 5))
pipe.fit(x_dev, y_dev)

random_results = pipe.named_steps["randomizedsearchcv"]

print("-------- GradientBoostingClassifier --------")
print("Best Score:", random_results.best_score_)
print("Best Params:", random_results.best_params_)
print("Best Score:", pipe.score(x_test, y_test))
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[39]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

start_time = time.time()

HGBC = HistGradientBoostingClassifier()

model_params = {
    'max_iter': random.sample(range(1, 200), 3),
    'max_depth': random.sample(range(1, 10), 3),
    'learning_rate': [0.001, 0.01, 0.1]
}

pipe = pipeline.make_pipeline(RandomizedSearchCV(HGBC, model_params, return_train_score = True, cv = 5, verbose = 5))
pipe.fit(x_dev, y_dev)

random_results = pipe.named_steps["randomizedsearchcv"]
print("-------- HistGradientBoostingClassifier --------")
print("Best Score:", random_results.best_score_)
print("Best Params:", random_results.best_params_)
print("Best Score:", pipe.score(x_test, y_test))
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# **3.2: Train an XGBoost model by tuning 3 hyperparameters using 5 fold cross-validation. Compare the performance of the trained XGBoost model on the test dataset against the performances obtained from 3.1**

# From 3.1, GradientBoostingClassifier best test score is  0.7917243626293068 and HistGradientBoostingClassifier best test score is 0.7987293312698542. Meanwhile XGBoost best test score is 0.8014444353940974. 
# XGBoost performed better than both GradientBoostingClassifier and HistGradientBoostingClassifier.

# In[40]:


#!pip install xgboost


# In[41]:


from xgboost import XGBClassifier

start_time = time.time()

XGB = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model_params = {
    'n_estimators': random.sample(range(1, 200), 3),
    'max_depth': random.sample(range(1, 10), 3),
    'learning_rate': [0.001, 0.01, 0.1]
}

pipe = pipeline.make_pipeline(RandomizedSearchCV(XGB, model_params, return_train_score = True, cv = 5, verbose = 5))
pipe.fit(x_dev, y_dev)

random_results = pipe.named_steps["randomizedsearchcv"]
print("-------- XGBClassifier --------")
print("Best Score:", random_results.best_score_)
print("Best Params:", random_results.best_params_)
print("Best Score:", pipe.score(x_test, y_test))
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# **3.3: Compare the results on the test dataset from XGBoost, HistGradientBoostingClassifier, GradientBoostingClassifier with results from Q1.6 and Q2.1. Which model tends to perform the best and which one does the worst? How big is the difference between the two? Which model would you choose among these 5 models and why?**

# XGBoost best test score is 0.8014444353940974.
# HistGradientBoostingClassifier best test score is 0.7987293312698542.
# GradientBoostingClassifier best test score is 0.7917243626293068.
# Random Forest best test score is 0.7895251282886698.
# Decision Tree best test score is 0.7271863375960468.
# 
# The best model is XGBoost and the worst model is Decision tree. The difference between the two model score is around 0.074.
# 
# I would choose XGBoost among these models because its test score is highest and also, the model selection time does not take that long.

# **3.4: Can you list the top 3 features from the trained XGBoost model? How do they differ from the features found from Random Forest and Decision Tree? Which one would you trust the most?**

# In[42]:


XGB.n_estimators = 133
XGB.max_depth = 6
XGB.learning_rate = 0.1

XGB.fit(x_dev, y_dev)

feat_imp = XGB.feature_importances_

col_list = player_df_X.columns.values

k=3
index = np.argpartition(feat_imp, len(feat_imp) - k)[-k:]
print("Top 3 Most Important Features:", col_list[index])


# The top 3 features from Random Forest were : 'vlDamage', 'qtDeath', 'qtSurvived'.
# The top 3 features from Decision Tree were : 'idRoom', 'qtDeath', 'qtSurvived'.
# Hence, the top 3 features from XGBoost is the same as those of the Decision Tree.
# 
# I would trust the top 3 features of XGBoost most, because it has the highest score.

# **3.5: Can you choose the top 7 features (as given by feature importances from XGBoost) and repeat Q3.2? Does this model perform better than the one trained in Q3.2? Why or why not is the performance better?**

# In[43]:


feat_imp = XGB.feature_importances_

col_list = player_df_X.columns.values

k=7
index = np.argpartition(feat_imp, len(feat_imp) - k)[-k:]
print("Top 7 Most Important Features:", col_list[index])


# In[44]:


## Drop all features excluding the 7 features above (for both dev and test dataset)
temp_dev = pd.DataFrame(x_dev, columns = col_list)[col_list[index]]
print(temp_dev)


temp_test = pd.DataFrame(x_test, columns = col_list)[col_list[index]]
print(temp_test)


# In[45]:


start_time = time.time()

XGB_new = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model_params = {
    'n_estimators': random.sample(range(1, 200), 3),
    'max_depth': random.sample(range(1, 10), 3),
    'learning_rate': [0.001, 0.01, 0.1]
}

pipe = pipeline.make_pipeline(RandomizedSearchCV(XGB_new, model_params, return_train_score = True, cv = 5, verbose = 5))
pipe.fit(temp_dev, y_dev)

random_results = pipe.named_steps["randomizedsearchcv"]
print("-------- XGBClassifier (For Top 7 Features) --------")
print("Best Score:", random_results.best_score_)
print("Best Params:", random_results.best_params_)
print("Best Score:", pipe.score(temp_test, y_test))
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# **Question: Compare the performance of the trained XGBoost model on the test dataset against the performances obtained from 3.1**
# 
# **Answer:** From 3.1, GradientBoostingClassifier best test score is 0.7917243626293068 and HistGradientBoostingClassifier best test score is 0.7987293312698542. Meanwhile XGBoost best test score is 0.8014444353940974. XGBoost performed better than both GradientBoostingClassifier and HistGradientBoostingClassifier.
# 
# After dropping all features except the 7 most important ones, XGBoost has test score of 0.793624935516277, which is now inferior compared to the original models. This shows that having more features (even though they might not be the top n important ones) is cruical to having a robust and good model performance, because those features give richer dimensional information about the data.

# ## Question 4: Calibration

# **4.1: Estimate the brier score for the XGBoost model (trained with optimal hyperparameters from Q3.2) scored on the test dataset.**

# In[46]:


#conda update scikit-learn


# In[54]:


from sklearn.calibration import calibration_curve, CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss

model_score_test = XGB.predict_proba(x_test)
y_true_test_flatten = y_test.values.reshape(-1, 1)
y_pred_prob= np.take_along_axis(model_score_test, y_true_test_flatten, axis=1)
brier = brier_score_loss(y_true_test_flatten, y_pred_prob)
print("Brier Score:", brier)

disp = CalibrationDisplay.from_estimator(XGB, x_test, y_test, n_bins = 10)


# In[49]:


# split dev to train & val
x_train, x_calib, y_train, y_calib = train_test_split(x_dev, y_dev, test_size = 0.2, random_state=0)


# **4.2: Calibrate the trained XGBoost model using isotonic regression as well as Platt scaling. Plot predicted v.s. actual on test datasets from both the calibration methods**

# In[57]:


## Iso Scaling
XGB_iso_calib = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# set optimal hyper-param    
XGB_iso_calib.n_estimators = 161
XGB_iso_calib.max_depth = 8
XGB_iso_calib.learning_rate = 0.1

XGB_iso_calib.fit(x_train, y_train)
disp = CalibrationDisplay.from_estimator(XGB_iso_calib, x_test, y_test, n_bins = 10, name="uncalibrated model w/ best hyperparam")

calibrated_model_iso = CalibratedClassifierCV(XGB_iso_calib, cv="prefit", method="isotonic")
calibrated_model_iso.fit(x_calib, y_calib)
display = CalibrationDisplay.from_estimator(calibrated_model_iso, x_test, y_test, n_bins = 10, name="calibrated model (Iso)")


# In[56]:


## Platt Scaling
XGB_platt_calib = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# set optimal hyper-param    
XGB_platt_calib.n_estimators = 161
XGB_platt_calib.max_depth = 8
XGB_platt_calib.learning_rate = 0.1

XGB_platt_calib.fit(x_train, y_train)
disp = CalibrationDisplay.from_estimator(XGB_platt_calib, x_test, y_test, n_bins = 10, name="uncalibrated model w/ best hyperparam")

calibrated_model_platt = CalibratedClassifierCV(XGB_platt_calib, cv="prefit", method="sigmoid")
calibrated_model_platt.fit(x_calib, y_calib)
display = CalibrationDisplay.from_estimator(calibrated_model_platt, x_test, y_test, n_bins = 10, name="calibrated model (Platt)")


# **4.3: Report brier scores from both the calibration methods. Do the calibration methods help in having better predicted probabilities?**

# In[58]:


model_score_test = calibrated_model_iso.predict_proba(x_test)
y_true_test_flatten = y_test.values.reshape(-1, 1)
y_pred_prob= np.take_along_axis(model_score_test, y_true_test_flatten, axis=1)
brier = brier_score_loss(y_true_test_flatten, y_pred_prob)
print("Brier Score:", brier)


# In[59]:


model_score_test = calibrated_model_platt.predict_proba(x_test)
y_true_test_flatten = y_test.values.reshape(-1, 1)
y_pred_prob= np.take_along_axis(model_score_test, y_true_test_flatten, axis=1)
brier = brier_score_loss(y_true_test_flatten, y_pred_prob)
print("Brier Score:", brier)


# The calibration theoretically should help decrease Brier Score and help in having better predicted probabilities, but in our example, because the dataset is already well calibrated, applying the two calibration method doesn't show much positive effect.

# In[ ]:




