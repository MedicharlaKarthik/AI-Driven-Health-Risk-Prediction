# AI-Driven Health Risk Prediction For Interconnected Chronic Diseases

This is a Machine Learning application built using **Streamlit**.
## 👨‍💼 Team Leadership

This is Major project of my B.Tech,I served as the **Team Leader** successfully completed and submitted to College as part of Academics.
 It predicts the risk levels for:
- Diabetes
- Heart Disease
- Kidney Disease (via CT scan images)

Users can input personal health details or upload kidney scan images, then it provides risk predictions for each disease individually. A final overall health risk is then calculated.

---



---

## 🔍 Project Overview

### 🧪 Diseases Include:
- **Heart Disease**
- **Diabetes**
- **Kidney Disease**

### 📝 Input Methods:
- **Heart & Diabetes**: Text input fields
- **Kidney**: CT Scan image upload

### ✅ Output Predictions:
- **Diabetes**: Yes / No  
- **Heart Disease**: Yes / No  
- **Kidney Disease**: Cyst / Normal / Stone / Tumor

### 📊 Final Health Risk Evaluation:
- **High Risk**: All 3 diseases detected
- **Medium Risk**: 1 or 2 diseases detected
- **No Risk**: None detected

---

## 🤖 Machine Learning Models Used


| Disease        | Algorithms Tested                                                                                       | Final Model Selected | Accuracy           |
|----------------|--------------------------------------------------------------------------------------------------------|----------------------|--------------------|
| **Diabetes**   | Logistic Regression (95%), Random Forest (96%)                                                         | Random Forest         | **96%**            |
| **Heart**      | K-Nearest Neighbors (87%), Random Forest (84%), SVM (83%), Decision Tree (79%)                         | K-Nearest Neighbors   | **87%** (with K=8) |
| **Kidney**     | Convolutional Neural Network (Sequential CNN with convolutional and dense layers)                      | CNN                   | **99%**            |
| **Final Risk** | Support Vector Machine (used to combine 3 disease predictions into overall health risk classification) | SVM                   | **75%**              |      

---

## 🛠️ Tech Stack

- **Frontend & Interface**: Streamlit
- **Languages**: Python
- **Libraries**: - `scikit-learn` for ML models  
  - `pandas`, `numpy` for data processing  
  - `Keras`, `OpenCV` for kidney image classification  
  - `Jupyter Notebooks` for training and experimentation
 
### Below are some screenshots of the user interface and prediction output:

#### Diabetes Input Form
![Diabetes Input Form](screenshots/diabetes_input_form.jpg)

#### Diabetes Prediction Output
![Diabetes Prediction](screenshots/diabetes_pred.jpg)

#### Heart Input Form
![Heart Input Form](screenshots/heart_input_form.jpg)

#### Heart Prediction Output
![Heart Prediction](screenshots/heart_prediction.jpg)

#### Kidney Input Form
![Kidney Input Form](screenshots/kidney_input_form.jpg)

#### Kidney Input Image
![Kidney Input Image](screenshots/kidney_input_img.jpg)

#### Overall Health Risk Prediction
![Overall Health Risk](screenshots/overall_health_risk.jpg)

