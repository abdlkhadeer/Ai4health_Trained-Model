#!/usr/bin/env python
# coding: utf-8

# In[24]:


# import data loading and visualization libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# confusion_matrix is in sklearn.metrics, not sklearn.linear_model
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


# import warnings
import warnings
warnings.filterwarnings("ignore")


# In[25]:


df = pd.read_csv('Training.csv')

# Display the first few rows of the dataframe
df.head()


# In[26]:


df.tail()


# In[27]:


df.shape


# In[28]:


len(df['treatment'].unique())


# In[29]:


# Split The Data in Train-Test Split

X = df.drop(columns=['treatment'])  # Features
y = df['treatment']  # Target variable


# In[30]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[31]:


# Split data into raining and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) 


# ## Defining a function to evaluate models and display metrics

# In[32]:


# Defining a function to evaluate models and display metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)  
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)
    
    # Calculate ROC AUC for multi-class
    try:
        if y_pred_proba is not None:
            unique_classes = np.unique(y_test)
            if len(unique_classes) > 2:
                roc_auc = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 2)  
            else:
                roc_auc = round(roc_auc_score(y_test, y_pred_proba[:, 1]), 2) 
        else:
            roc_auc = "Not Applicable"
    except Exception as e:
        roc_auc = "Not Calculable"
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# ## Confusion Matrix Plotting Function

# In[33]:


def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# ## ROC Curve Plotting Function

# In[34]:


def plot_roc_curve(y_test, y_pred_proba, model_name):
    try:
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 2:  # Multi-class ROC
            plt.figure(figsize=(8, 6))
            for i in range(len(unique_classes)):
                fpr, tpr, _ = roc_curve(y_test == unique_classes[i], y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {unique_classes[i]} (AUC = {roc_auc_score(y_test, y_pred_proba, multi_class="ovr"):.2f})')
        else:  # Binary ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba[:, 1]):.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.show()
    except Exception as e:
        print(f"ROC curve could not be plotted for {model_name}: {e}")


# ## Evaluating The Multiple Models

# In[35]:


# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100),
    "SVC" : SVC(kernel='linear', random_state=42, probability=True) 

}


# Initialize an empty dictionary to store results
results_dict = {}

# Loop through the models and evaluate each
for model_name, model in models.items():
    print(f"\nEvaluating: {model_name}")
    
    # Train and evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
     # Store metrics in the results dictionary
    results_dict[model_name] = metrics # Store metrics for each model
    
    # Display metrics
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    
    # Plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Plot ROC curve if applicable
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        plot_roc_curve(y_test, y_pred_proba, model_name)
    else:
        print(f"ROC curve not available for {model_name}")


# In[20]:


# Create DataFrame directly from the results_dict
results = pd.DataFrame.from_dict(results_dict, orient='index')

# Resetting index to make 'Model' a column
results = results.reset_index().rename(columns={'index': 'Model'})

results_sorted = results.sort_values(by='roc_auc', ascending=False) 

# Display the comparison table
print("\nModel Comparison:")
print(results_sorted)


# ## Bar Plot Showing The Comparison Bewtween Models and Evaluation Metric

# In[36]:


# Bar Plot for results comparison 
results_sorted.plot(x='Model', y=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], kind='bar', figsize=(10,6))
plt.title('Model Performance Comparison')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.show()


# ## Performing Hyperparameter Tuning To Get The Best Parameter For Each Model 

# In[37]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],  
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.6, 0.8, 1.0]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'], 
        'gamma': ['scale', 'auto']  
    }
}


# In[38]:


# Initializing an empty dictionary to store the best models and their parameters
best_models = {}

# Loop through each model and apply GridSearchCV
for model_name, model in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grids[model_name], 
                               scoring='accuracy',  
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1)  
    
    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)
    
    # Store the best model and its best parameters
    best_models[model_name] = grid_search.best_estimator_
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_:.2f}")


# In[39]:


# Re-evaluating the XGBoost model as the best model selection 
def re_evaluate_xgboost(best_xgb_model, X_test, y_test):
    # Predict on the test set
    y_pred = best_xgb_model.predict(X_test)
    y_pred_proba = best_xgb_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Check if multiclass and calculate ROC AUC
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.show()

    # Plot ROC Curve
    fpr = {}
    tpr = {}
    for i, class_label in enumerate(unique_classes):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test == class_label, y_pred_proba[:, i])
        plt.plot(fpr[class_label], tpr[class_label], label=f'Class {class_label} ROC (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Positioning the legend outside the plot for clearer visualization
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    
    plt.tight_layout()  
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Get the best XGBoost model from the dictionary
best_xgb_model = best_models['XGBoost']

# Re-evaluate the tuned XGBoost model
print("\nRe-evaluation of Tuned XGBoost Model:")
xgb_metrics = re_evaluate_xgboost(best_xgb_model, X_test, y_test)

# Print the re-evaluated metrics
print("\nRe-evaluated Metrics for Tuned XGBoost:")
print(f"Accuracy: {xgb_metrics['accuracy']:.2f}")
print(f"Precision: {xgb_metrics['precision']:.2f}")
print(f"Recall: {xgb_metrics['recall']:.2f}")
print(f"F1 Score: {xgb_metrics['f1']:.2f}")
print(f"ROC AUC: {xgb_metrics['roc_auc']:.2f}")


# ## Saving the model

# In[40]:


# using joblib to save XGBoost model
import joblib

joblib.dump(best_xgb_model, 'xgb_model.pkl')


# In[44]:


# pred on the 2d array above to check if our model pred correctly or not

# test 1 :
print('Model Predictions :',best_xgb_model.predict(X_test.iloc[0].values.reshape(1,-1)))
print('Actual Labels :', y_test[0])


# In[54]:


# test 2 :
print('Model Predictions :',best_xgb_model.predict(X_test.iloc[40].values.reshape(1,-1)))
print('Actual Labels :', y_test[40])

