# %%
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score


# %%
df = pd.read_csv('winequality-red.csv')
x = df.drop('quality', axis=1)
y = df['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(penalty='none', max_iter=100)
lr.fit(x_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(x_test))
auc = roc_auc_score(y_test, lr.predict_proba(x_test), multi_class='ovr')
print(f"Accuracy Score of Training Data is {accuracy_score(y_train, lr.predict(x_train))}")
print(f"Accuracy Score of Training Data is {lr_acc}\n")
print(f"AUC of model is {auc}\n")


# %%
