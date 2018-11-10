from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import os
cwd = os.getcwd()

### Read in and set data
eeg = pd.read_csv(os.path.join(cwd, 'EEGEyeState.csv'))

X = eeg.ix[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13)].values
y = eeg.ix[:,14].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
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


## ROC and AUC
# Find ROC and AUC of train set
y_probs_train = tree_classf.predict_proba(X_train)
y_preds_train = y_probs_train[:,1]
fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_preds_train)
roc_auc_train = auc(fpr_train, tpr_train)

# Find ROC and AUC of test set
y_probs_test = tree_classf.predict_proba(X_test)
y_preds_test = y_probs_test[:,1]
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_preds_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot it all
plt.title('ROC Curve')
plt.plot(fpr_train, tpr_test, 'orange', label = 'Train: AUC = %0.2f' % roc_auc_train)
plt.plot(fpr_test, tpr_test, 'b', label = 'Test: AUC = %0.2f' % roc_auc_test)
plt.plot()
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
