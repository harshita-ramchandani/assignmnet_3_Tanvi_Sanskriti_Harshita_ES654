# DESCRIPTION
\
\
## Dataset
The dataset has been created using sklearn's make_classification() and has 100 samples and 20 features as we were asked to create a high dimensional datatset
\
\
**Number of samples:** 100
\
\
**Number of features:** 20
\
\
**Number of classes:** 2(as it is a binary classification problem)
\
\
## Model
\
\
We are using **Logistic Regression** and **Linear Regression** as our classifiers.
### Working of our code
Our code generates a synthetic dataset using make_classification from scikit-learn, initializes two SequentialFeatureSelector objects for forward and backward feature selection, and loops through a range of iterations to select a variable number of features. At each iteration, our code fits the SequentialFeatureSelector object, gets the selected features and their corresponding scores, appends the number of selected features and the corresponding score to the lists, and plots the scores using matplotlib. It saves the selected models using pickle. Finally, we are displaying the app in Streamlit and we have created a slider that allows the user to interact with the plot by selecting the number of iterations to display.





### Interpretation of the graphs
*Logistic Regression*
In the Logistic Regression graph as we can clearly see that with the increase in number of iterations, the accuracy score is improving.Forward feature selection and backward feature selection both are converging to almost similar values. But Forward feature selection seems to perform better with increased number of informative fatures selected.



### First Plot:
*Linear Regression* 
Forward Feature Selection

The first plot is for forward feature selection and uses a slider to change the number of selected features. The plot shows the ***mean squared error (MSE)*** for two lines:

The MSE using all features *(in blue)*
The MSE using only the selected features *(in red dots)*
The x-axis shows the number of selected features, and the y-axis shows the MSE. 


### Second Plot: 
**Forward and Backward Feature Selection**
The second plot is for both forward and backward feature selection and uses a slider to change the number of selected features. The plot shows the accuracy score for two lines:

The accuracy using forward selection *(in blue)*
The accuracy using backward selection *(in orange)*
The x-axis shows the number of selected features, and the y-axis shows the accuracy score. The plot updates as the slider value changes, showing how the accuracy changes as more features are selected in both the forward and backward directions. The plot is interactive and allows the user to change the number of iterations with the slider, so they can see the effect of selecting more or fewer features on the accuracy scores.
