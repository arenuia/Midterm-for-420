from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import os

cwd = os.getcwd()
eeg = pd.read_csv(os.path.join(cwd, 'EEGEyeState.csv'))

X = eeg.ix[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13)].values
y = eeg.ix[:,14].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
tree_classf = RandomForestClassifier(n_estimators=10)
tree_classf.fit(X_train, y_train)
y_probs = tree_classf.predict_proba(X_test)
y_preds = y_probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
