# DS_capstone
# Diabetes Prediction using Deep Learning
# Project Overview:
This project aims to predict the likelihood of an individual having diabetes (pre-diabetes, Type 2 diabetes, or gestational diabetes) using a deep learning model. The model is built using a feedforward neural network (FNN) and trained on the Pima Indians Diabetes Dataset. The goal is to create a user-friendly, reliable prediction system for healthcare professionals and individuals.
# Dataset:
The dataset used in this project is the Pima Indians Diabetes Dataset, which includes 768 observations with 8 features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skinfold thickness (mm)
Insulin: 2-hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
Diabetes Pedigree Function: A function that accounts for genetic predisposition to diabetes
Age: Age in years
The target variable is Outcome, where 1 indicates diabetes and 0 indicates no diabetes.
 # Data Prepossing:
 Before training, the following steps were applied:

Handling Missing Values: Columns such as Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI contained zero values, which were treated as missing data and imputed using the mean/median values.
Scaling: Features were standardized using StandardScaler to ensure uniformity.
Train-Test Split: The dataset was split into 70% training and 30% testing sets

# Model Architechture:
A feedforward neural network (FNN) with the following structure was used:

Input Layer: 8 neurons corresponding to the 8 input features.
Hidden Layers:
First hidden layer with 16 neurons and ReLU activation.
Second hidden layer with 8 neurons and ReLU activation.
Output Layer: 1 neuron with a sigmoid activation function for binary classification (diabetes or not).
The model was compiled using:

Loss Function: binary_crossentropy (for binary classification).
Optimizer: Adam (for adaptive learning).
Metrics: Accuracy and validation accuracy

# Training
The model was trained for 50 epochs using a batch size of 32. Early stopping and model checkpointing were implemented to prevent overfitting and save the best model during validation.

# Evaluation:
Accuracy: Evaluate the percentage of correctly classified cases.
Precision, Recall, and F1-Score: Provide deeper insight into model performance, especially for handling class imbalance.
Confusion Matrix: Used to visualize correct and incorrect predictions for each class.

# Results:
Test Accuracy: ~78% accuracy on the test set.
Precision/Recall: Precision and recall metrics were used to measure the model’s performance in detecting true positives (diabetes cases) and minimizing false positives/negatives.
Confusion Matrix: Demonstrates a balanced classification performance between diabetic and non-diabetic cases.

# Deployment:
The model was deployed using Streamlit, creating a user-friendly web application where users can input their health data and receive real-time diabetes predictions.

# How to Use code
1. Clone the project repository to local maachine
2. Navigate to project directory.
3. Install the Required Dependencies from "requirments.txt"
4. Download the Dataset from "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database".
5. Run the code in "jupyter notebook" or any other Python IDE.
6. Run the Streamlit web application.

# How this model could be applied or adapted to your ongoing project.
* Predicting and Classifying Diabetes Types (Type 2 or Gestational)
  The current model predicts diabetes presence (binary classification). We can modify the model to a multi-class classifier to predict different diabetes types (e.g., Type 2 vs. Gestational Diabetes) using similar health and biometric data but adapting the output layer and using categorical cross-entropy as the loss function.






