import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.ensemble import (
    AdaBoostClassifier
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AdaBoost Classification",
    layout="wide"
)

st.title(
    "Glass Classification using AdaBoost"
)

st.write(
    "AdaBoost Classification with Hyperparameter Tuning"
)

os.makedirs(
    "models",
    exist_ok=True
)

# =====================================================
# Load Dataset
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/glass.csv"
    )

    df.dropna(inplace=True)

    return df


df = load_data()

# =====================================================
# Dataset Preview
# =====================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Dataset Shape")

st.write(df.shape)

# =====================================================
# Features & Target
# =====================================================

X = df.drop(
    "Type",
    axis=1
)

y = df["Type"]

# =====================================================
# Sidebar Settings
# =====================================================

st.sidebar.header(
    "Model Settings"
)

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.5,
    0.2,
    0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    value=42
)

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# =====================================================
# Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

# =====================================================
# Hyperparameter Tuning
# =====================================================

st.subheader(
    "Hyperparameter Tuning"
)

base_estimator = DecisionTreeClassifier(
    max_depth=1,
    random_state=random_state
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0]
}

ada = AdaBoostClassifier(
    estimator=base_estimator,
    random_state=random_state
)

grid_search = GridSearchCV(
    ada,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

with st.spinner(
    "Training AdaBoost Model..."
):

    grid_search.fit(
        X_train_scaled,
        y_train
    )

best_model = grid_search.best_estimator_

pickle.dump(
    best_model,
    open(
        "models/adaboost_model.pkl",
        "wb"
    )
)

# =====================================================
# Best Parameters
# =====================================================

st.success(
    "Training Completed"
)

st.subheader(
    "Best Parameters"
)

st.write(
    grid_search.best_params_
)

st.subheader(
    "Best Cross Validation Accuracy"
)

st.write(
    f"{round(grid_search.best_score_ * 100, 2)} %"
)

# =====================================================
# Predictions
# =====================================================

y_pred = best_model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# =====================================================
# Accuracy
# =====================================================

st.subheader(
    "Model Accuracy"
)

st.write(
    f"{round(accuracy * 100, 2)} %"
)

# =====================================================
# Classification Report
# =====================================================

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose()
)

# =====================================================
# Confusion Matrix
# =====================================================

st.subheader(
    "Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(6,5)
)

ax.imshow(cm)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center"
        )

st.pyplot(fig)

# =====================================================
# Prediction Section
# =====================================================

st.subheader(
    "Predict Glass Type"
)

input_data = []

for col in X.columns:

    value = st.number_input(
        f"Enter {col}",
        value=float(
            X[col].mean()
        )
    )

    input_data.append(value)

if st.button("Predict"):

    scaler = pickle.load(
        open(
            "models/scaler.pkl",
            "rb"
        )
    )

    model = pickle.load(
        open(
            "models/adaboost_model.pkl",
            "rb"
        )
    )

    input_array = np.array(
        input_data
    ).reshape(1, -1)

    input_scaled = scaler.transform(
        input_array
    )

    prediction = model.predict(
        input_scaled
    )

    st.success(
        f"Predicted Glass Type: {prediction[0]}"
    )