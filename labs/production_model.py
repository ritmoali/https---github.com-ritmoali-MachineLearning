import pandas as pd
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib


test_sample = pd.read_csv("../labs/test_sample.csv")
model = joblib.load('best_model:pk1')

# Gör predictioner

X_test_samples = test_sample.drop('cardio', axis=1)
y_test_samples = test_sample['cardio']
probabilities = model.predict_proba(X_test_samples)
predictions = model.predict(X_test_samples)