# AI-Powered Skincare Recommendation System

https://corium.streamlit.app/

An interactive Machine Learning web application that provides personalized skincare product recommendations and daily skincare routines based on a user's skin profile, concerns, and severity levels.

Built using Streamlit and Scikit-Learn, the application demonstrates the complete machine learning workflow including dataset generation, data preprocessing, model training, evaluation, and deployment.

## Features

* Personalized skincare recommendations
* Skin health score calculation
* Morning and night routine generation
* Recommendation history tracking
* Downloadable PDF reports
* Interactive ML Insights dashboard
* Random Forest based recommendation engine
* Synthetic dataset generation for model training

## Machine Learning Workflow

### 1. Dataset Generation

A synthetic skincare dataset containing 1,200+ records is generated using different skin types, concerns, severity levels, and recommended skincare products.

### 2. Data Preprocessing

* Data cleaning
* Feature encoding
* Label transformation
* Training and testing split

### 3. Model Training

A Random Forest model is trained to predict suitable skincare products based on:

* Age
* Skin Type
* Skin Concern
* Severity Score

### 4. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Feature Importance Analysis

### 5. Recommendation Generation

The trained model predicts suitable:

* Cleanser
* Moisturizer
* Serum
* Sunscreen

The recommendations are then combined with rule-based routine generation to create personalized skincare plans.

## Tech Stack

### Frontend

* Streamlit

### Machine Learning

* Scikit-Learn
* Pandas
* NumPy
* Joblib

### Visualization

* Matplotlib

### Reporting

* ReportLab

## Project Structure

```text
skincare-recommendation/
│
├── data/
├── ml/
├── models/
├── pages/
├── products/
├── tests/
├── utils/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-skincare-recommendation-system.git
cd ai-skincare-recommendation-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python ml/generate_dataset.py
```

Train the model:

```bash
python ml/train_model.py
```

Run the application:

```bash
streamlit run app.py
```

## Future Improvements

* Real skincare product database integration
* User authentication and profiles
* Skin image analysis using computer vision
* Deep learning recommendation models
* Cloud deployment and API integration

## Author

Gaurang Kishore

Developed to help people better understand their skin concerns and receive personalized skincare recommendations, while applying machine learning techniques to create an accessible and user-friendly skincare guidance system.
