# multiclass_obesity_level_classification

**Obesity Level Classification**

**Project Overview**

This project explores and models obesity levels in individuals from Mexico, Peru, and Colombia using demographic, dietary, and physical activity data. The goal is to perform multiclass classification of obesity status using traditional machine learning and deep learning approaches, while applying appropriate preprocessing, exploratory analysis, and model evaluation techniques.

The target variable, NObeyesdad, consists of seven balanced classes:
	•	Insufficient Weight
	•	Normal Weight
	•	Overweight Level I
	•	Overweight Level II
	•	Obesity Type I
	•	Obesity Type II
	•	Obesity Type III

Class balance was achieved prior to analysis using SMOTE by the original dataset authors.
Dataset

The dataset includes:
	•	Categorical features (e.g., gender, eating habits)
	•	Numerical features (e.g., age, height, weight, physical activity metrics)
Data preprocessing and analysis are performed entirely within the Jupyter Notebook.

**Methods and Workflow**
1. Data Preprocessing
	•	Label encoding of categorical variables
	•	Merging categorical and numerical features
	•	Feature scaling applied only to numerical features using StandardScaler

2. Exploratory Data Analysis (EDA)
	•	Class distribution analysis of NObeyesdad
	•	Visualization of categorical variables
	•	Distribution analysis of numerical variables
	•	Q–Q plots and numerical skewness/kurtosis assessment

Key findings:
	•	Height and weight are approximately normally distributed
	•	Age is positively skewed
	•	Several numerical features exhibit skewness and kurtosis
  
Identified log-transformable features:
	•	Age
	•	FCVC
	•	NCP
	•	CH2O
	•	FAF
	•	TUE
Note: Logistic regression and neural networks do not strictly require normality

4. Feature Selection
	•	Correlation analysis using visual inspection
	•	No problematic multicollinearity detected among numerical features
	•	Final feature set retained without elimination

6. Models Implemented
The following models were trained and evaluated:
	•	One-vs-Rest Logistic Regression
	•	Base model
	•	Hyperparameter-tuned model
	•	Sequential Neural Network (Keras / TensorFlow)
	•	Base model
	•	Hyperparameter-tuned model

8. Model Evaluation
	•	Performance comparison across models
	•	One-vs-Rest logistic regression offers interpretability
	•	Neural network achieved best overall performance, but lacks explainability

10. Model Selection and Saving
	•	The tuned neural network was selected as the final model
	•	Best-performing model is saved for downstream use

**Technologies Used**
	•	Python
	•	Pandas, NumPy
	•	Matplotlib, Seaborn
	•	Scikit-learn
	•	TensorFlow / Keras
	•	Statsmodels
  
Notes
	•	This project focuses on predictive performance rather than causal inference
	•	Explainability is limited for the neural network model
	•	Suitable as a foundation for public health surveillance or decision-support tools
