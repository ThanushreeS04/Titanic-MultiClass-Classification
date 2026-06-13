# Titanic-MultiClass-Classification
# 🚢 Titanic Age Group Classification & Analysis

> A complete Machine Learning project that transforms the famous Titanic dataset into a **Multiclass Age Group Classification Problem**, combined with passenger behavior analysis, feature engineering, model comparison, clustering, and an interactive Streamlit deployment.

---

## 📖 Project Overview

The Titanic dataset is commonly used for binary survival prediction. In this project, the problem is reformulated as a **multiclass classification task**, where passengers are classified into age groups based on demographic and travel-related attributes.

The project investigates how factors such as **gender, passenger class, fare, family size, and survival status** relate to different age groups and evaluates multiple machine learning models to identify the most effective classifier.

---

## 🎯 Objectives

### Primary Objective
Classify Titanic passengers into age categories:

| Group | Age Range |
|---------|-----------|
| G1 | Age < 15 |
| G2 | 16 – 30 |
| G3 | 31 – 55 |
| G4 | 56 – 80 |
| G5 | Age > 80 |

### Secondary Objectives

- Analyze the relationship between passenger age groups and:
  - Gender
  - Passenger Class
  - Survival Status
- Study the relationship between:
  - Fare vs Passenger Class
  - Fare vs Survival Status
- Compare multiple machine learning algorithms.
- Perform clustering analysis.
- Deploy the best-performing model using Streamlit.

---

# 📂 Dataset

**Dataset:** Titanic Passenger Dataset

### Features Used

| Feature | Description |
|----------|-------------|
| Survived | Survival Status |
| Pclass | Passenger Class |
| Sex | Gender |
| Age | Passenger Age |
| SibSp | Siblings/Spouses Aboard |
| Parch | Parents/Children Aboard |
| Fare | Ticket Fare |
| Embarked | Port of Embarkation |
| Cabin | Cabin Number |
| Ticket | Ticket Number |
| Name | Passenger Name |

---

# ⚙️ Data Preprocessing

The dataset contained missing values and categorical attributes that required preprocessing.

### Missing Value Handling

| Column | Method |
|----------|----------|
| Age | Median Imputation |
| Embarked | Mode Imputation |
| Cabin | Unknown Category |

### Data Cleaning

- Removed irrelevant information where necessary.
- Converted categorical attributes into machine-readable format.
- Standardized numerical features.

---

# 🛠 Feature Engineering

To improve predictive performance, several new features were created:

### Family Features

```python
FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0
```

### Name-Based Features

- Extracted passenger titles:
  - Mr
  - Mrs
  - Miss
  - Master
  - Rare Titles

### Cabin-Based Features

- Extracted Deck information:
  - A, B, C, D, E, F, G
  - U (Unknown)

### Fare-Based Features

```python
FarePerPerson = Fare / FamilySize
```

### Ticket-Based Features

- TicketGroup
- Shared Ticket Identification

---

# 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

### Passenger Analysis

- Age Distribution
- Age Group Distribution
- Gender Distribution
- Passenger Class Distribution

### Survival Analysis

- Survival vs Gender
- Survival vs Passenger Class
- Survival vs Family Size

### Fare Analysis

- Fare vs Passenger Class
- Fare vs Survival Status
- Fare Distribution across Age Groups

### Visualizations

✔ Histograms  
✔ Countplots  
✔ Boxplots  
✔ Bar Charts  
✔ Correlation Heatmaps  
✔ Cluster Visualizations

---

# 🤖 Machine Learning Models

The following multiclass classification algorithms were implemented:

### 1. Logistic Regression

- Multinomial Classification
- Baseline Model

### 2. Random Forest Classifier

- Ensemble Learning
- Feature Importance Analysis

### 3. K-Nearest Neighbors (KNN)

- Distance-Based Classification

### 4. Gradient Boosting Classifier

- Sequential Ensemble Learning

### 5. Support Vector Machine (SVM)

- RBF Kernel

---

# 📈 Model Evaluation

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Stratified 5-Fold Cross Validation

## Cross Validation Results

| Model | Accuracy | F1 Score | Precision | Recall |
|---------|---------|---------|---------|---------|
| Logistic Regression | 62.77% | 61.09% | 60.22% | 62.77% |
| Gradient Boosting | 60.81% | 59.26% | 58.85% | 60.81% |
| KNN | 59.69% | 58.15% | 57.75% | 59.69% |
| Random Forest | 58.42% | 57.34% | 56.88% | 58.42% |
| SVC | 56.31% | 49.13% | 47.11% | 56.31% |

---

# 🏆 Best Performing Model

### Logistic Regression

**Performance:**

- Accuracy: 62.77%
- Precision: 60.22%
- Recall: 62.77%
- F1 Score: 61.09%

This model provided the best balance between performance and interpretability.

---

# 🔍 Feature Importance Analysis

Feature importance was analyzed to understand which attributes contributed most to age-group prediction.

Important predictors included:

- Passenger Class
- Fare
- Gender
- Family Size
- Survival Status
- Embarked Port

This helped explain the influence of socio-economic status and demographic characteristics on age-group classification.

---

# 🎯 Clustering Analysis

K-Means Clustering was applied as an additional exploratory task.

### Objectives

- Discover hidden passenger groups.
- Compare clusters with survival patterns.
- Analyze similarities among passengers.

### Techniques Used

- K-Means Clustering
- Elbow Method
- Silhouette Score

---

# 🚀 Streamlit Web Application

An interactive web application was developed using Streamlit.

### Features

✅ Passenger Data Input

✅ Real-Time Prediction

✅ Automatic Feature Engineering

✅ Age Group Classification

✅ User-Friendly Interface

---

## Running the Application

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Streamlit

```bash
python -m streamlit run streamlit_app.py
```

### Open Browser

```text
http://localhost:8501
```

---

# 📁 Project Structure

```text
Titanic-AgeGroup-Classification/
│
├── data/
│   └── train.csv
│
├── notebooks/
│   └── Titanic_AgeGroup_Classification.ipynb
│
├── models/
│   └── titanic_agegroup_pipeline.joblib
│
├── streamlit_app.py
│
├── requirements.txt
│
├── README.md
│
├── report.pdf
│
└── images/
    ├── age_distribution.png
    ├── survival_analysis.png
    ├── model_comparison.png
    └── clustering_results.png
```

---

# 💡 Future Improvements

- Hyperparameter Tuning
- XGBoost / CatBoost Integration
- Deep Learning Models
- SHAP Explainability
- SMOTE for Class Balancing
- Cloud Deployment
- Interactive Dashboard Enhancements

---

# 📚 Tech Stack

| Category | Tools |
|-----------|--------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Deployment | Streamlit |
| Model Persistence | Joblib |

---

# 🎓 Learning Outcomes

Through this project, the following concepts were explored:

- Data Preprocessing
- Feature Engineering
- Multiclass Classification
- Cross Validation
- Model Evaluation
- Clustering
- Data Visualization
- Streamlit Deployment
- End-to-End ML Workflow

---

# 👩‍💻 Author

**Thanushree Sudhakar**

B.Tech – Computer Science & Engineering  
Specialization: AI & Data Science / IoT  
REVA University

---

⭐ If you found this project useful, consider giving it a star on GitHub!
