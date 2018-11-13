from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sn

import os
cwd = os.getcwd()

### Read in and set data
eeg = pd.read_csv(os.path.join(cwd, 'EEGEyeState.csv'))

X = eeg.ix[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13)].values
y = eeg.ix[:,14].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=420)
# n_estimators = 100
tree_classf = RandomForestClassifier(n_estimators=10)
tree_classf.fit(X_train, y_train)


### Validation

## Classification Accuracy
y_bipreds_test = tree_classf.predict(X_test)
metrics.accuracy_score(y_test, y_bipreds_test)
# Model Accuracy Score: 0.8936359590565198

# Null Accuracy (Accuracy given that we predict the most frequent class every time)
max(y_test.mean(), 1 - y_test.mean())
# Null Accuracy Score: 0.5505117935024477

# Analysis:
# We get better accuracy given that we use this Random Forest model, with an
# increase of accuracy of 34.312%.


## Confusion Matrix
# On Test Data
cf_mX_test = metrics.confusion_matrix(y_test, y_bipreds_test)
cm_df_test = pd.DataFrame(cf_mX_test, range(2), range(2))
# Graphing Confusion Matrix
aX_test = sn.heatmap(cm_df_test, annot=True, cmap='Blues', fmt='g')
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
aX_test.xaxis.tick_top()
plt.savefig("Conf_Matrix.png")

## ROC and AUC
# Find ROC and AUC of n_estimators
y_probs_test = tree_classf.predict_proba(X_test)
y_preds_test = y_probs_test[:,1]
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_preds_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot it all
plt.title('Default Parameters: ROC Curve')
plt.plot(fpr_test, tpr_test, 'b', label = 'AUC = %0.2f' % roc_auc_test)
plt.plot()
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("ROC_Curve.png")
plt.show()


## Parameter Testing
# Copied from https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d #
# ROC Optimization
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.title("N of Estimators Variation")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.savefig("NEstimatorsVar.png")
plt.show()

# Tree Depth Optimization
max_depths = np.linspace(1, 50, 100, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.title("Tree Depth Variation")
plt.savefig("DepthVariation.png")
plt.show()
