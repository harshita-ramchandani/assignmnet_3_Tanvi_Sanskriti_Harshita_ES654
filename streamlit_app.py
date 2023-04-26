import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from IPython.display import display
from joblib import Parallel, delayed
from tqdm import tqdm


Accuracycount={}
st.title("Logistic Regression")
st.subheader("Forward and Backward Feature Selection")
st.header("Machine Learning")
st.markdown("---  ")

X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           class_sep=2.0, random_state=42)

sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=10)

sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=10)

n_features_fwd = np.zeros(2)
scores_fwd = np.zeros(2)
n_features_bwd = np.zeros(2)
scores_bwd = np.zeros(2)

st.write('Starting a long computation...')
for i in tqdm(range(1, 3)):

    def fit_sfs(X, y, i):
        sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward', n_features_to_select=i)
        selected_features_fwd = sfs.fit_transform(X, y)
        lr = LogisticRegression().fit(selected_features_fwd, y)
        score_fwd = accuracy_score(y, lr.predict(selected_features_fwd))
        return score_fwd

    def fit_sbs(X, y, i):
        sbs = SequentialFeatureSelector(LogisticRegression(), direction='backward', n_features_to_select=i)
        selected_features_bwd = sbs.fit_transform(X, y)
        lr = LogisticRegression().fit(selected_features_bwd, y)
        score_bwd = accuracy_score(y, lr.predict(selected_features_bwd))
        return score_bwd

    scores_fwd[i-1] = fit_sfs(X, y, i)
    scores_bwd[i-1] = fit_sbs(X, y, i)
    n_features_fwd[i-1] = i
    n_features_bwd[i-1] = i

st.balloons()
st.success('Done!')

def plot_scores(n):
    fig=plt.figure()
    plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
    plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
    plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Sequential Feature Selection')
    plt.legend()
   
