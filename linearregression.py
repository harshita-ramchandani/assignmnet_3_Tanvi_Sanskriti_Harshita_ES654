from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from IPython.display import display
import numpy as np
import pandas as pd
import streamlit as st


st.title("Linear Regression")
st.subheader("Forward and Backward Feature Selection")
st.header("Machine Learning")
st.markdown("---")

X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                           n_redundant=2, n_repeated=0, n_classes=1, # changed n_classes to 1
                           class_sep=2.0, random_state=42)

# change y to continuous values
y = np.random.randn(len(y))

# create a sequential forward feature selector
sfs = SequentialFeatureSelector(LinearRegression(), direction='forward', n_features_to_select=10)

# create a sequential backward feature selector
sbs = SequentialFeatureSelector(LinearRegression(), direction='backward', n_features_to_select=10)

# initialize empty lists to store the number of selected features and corresponding scores
n_features_fwd = []
scores_fwd = []
n_features_bwd = []
scores_bwd = []

# loop through a range of iterations to select variable number of features
for i in range(1, 20):
    # create a new instance of the linear regression model
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(LinearRegression(), direction='forward', n_features_to_select=i) 
    # fit the sequential forward feature selector
    sfs.fit(X, y)
    # get the selected features and their corresponding scores
    selected_features_fwd = sfs.transform(X)
    lr.fit(selected_features_fwd, y)
    score_fwd = mean_squared_error(y, lr.predict(selected_features_fwd))
    # append the number of selected features and the corresponding score to the lists
    n_features_fwd.append(i)
    scores_fwd.append(score_fwd)
    #print(score_fwd)

    # create a new instance of the linear regression model
    lr = LinearRegression()
    sbs = SequentialFeatureSelector(LinearRegression(), direction='backward', n_features_to_select=i)
    # fit the sequential backward feature selector
    sbs.fit(X, y)
    # get the selected features and their corresponding scores
    selected_features_bwd = sbs.transform(X)
    lr.fit(selected_features_bwd, y)
    score_bwd = mean_squared_error(y, lr.predict(selected_features_bwd))
    # append the number of selected features and the corresponding score to the lists
    n_features_bwd.append(i)
    scores_bwd.append(score_bwd)
    #print(score_bwd)

# store the scores of forward and backward feature selection in a DataFrame
df_scores = pd.DataFrame({'backward': scores_bwd, 'forward': scores_fwd})
st.write(df_scores)

# define a function to plot the scores
def plot_scores(n):
    fig=plt.figure()
    plt.plot(n_features_fwd[:n], scores_fwd[:n], label='Forward')
    plt.plot(n_features_bwd[:n], scores_bwd[:n], label='Backward')
    plt.xlabel('Number of Features')
    plt.ylabel('MSE')
    plt.title('Sequential Feature Selection')
    plt.legend()
    #plt.show()
    st.pyplot(fig)

# create a slider for the number of iterations
#iterations_slider = IntSlider(min=1, max=len(n_features_fwd), value=len(n_features_fwd), description='Iterations:')
iterations_slider = st.slider(label="Iterations", min_value=1, max_value=len(n_features_fwd), value=len(n_features_fwd))

# use the interact function to link the
interact(plot_scores, n=iterations_slider)
