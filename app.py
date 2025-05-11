import sys
import numpy as np
import pandas as pd 
import streamlit as st
import joblib
import os
import cv2
import tensorflow as tf

# Fix for missing numpy._core issue
import numpy.core
sys.modules["numpy._core"] = numpy.core

# ---------------------- MODEL LOADING FUNCTIONS ----------------------
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path, mmap_mode='r')
    else:
        st.error(f"File not found: {model_path}")
        return None

# Load models for Diabetes
log_reg_model = load_model(r"models/diabetes_log_reg_model.pkl")
rf_model = load_model(r"models/diabetes_rf_model.pkl")
diabetes_scaler = load_model(r"models/diabetes_scaler.pkl")

# Load models for Heart Disease
knn_model = load_model(r"models/heart_knn_model.pkl")
svm_model = load_model(r"models/heart_svm_model.pkl")
dt_model = load_model(r"models/heart_dt_model.pkl")
rf_heart_model = load_model(r"models/heart_rf_model.pkl")
heart_scaler = load_model(r"models/heart_scaler.pkl")  # Added Heart Disease Scaler

# Load Kidney Disease Model using TensorFlow (H5 format)
@st.cache_resource
def load_kidney_model():
    return tf.keras.models.load_model("models/kidney_disease_model.h5")
kidney_model = load_kidney_model()

# ---------------------- ADDITIONAL INFORMATION ----------------------
diabetes_info = {
    "Patient is Diabetic": (
        "*Possible Causes:*\n"
        "- Genetic predisposition\n"
        "- Poor diet and high sugar intake\n"
        "- Lack of physical activity\n"
        "- Obesity and high BMI\n"
        "- High blood pressure and cholesterol levels\n\n"
        "*Precautions:*\n"
        "- Maintain a balanced, low-sugar diet\n"
        "- Engage in regular physical activity (at least 30 minutes daily)\n"
        "- Monitor blood glucose and HbA1c levels regularly\n"
        "- Avoid smoking and limit alcohol consumption\n\n"
        "*Life Risk:*\n"
        "- High risk if left untreated; can lead to kidney failure, nerve damage, and cardiovascular diseases."
    ),
    "Patient is Non-Diabetic": (
        "- No diabetes detected. Maintain a healthy lifestyle!"
    )
}

heart_info = {
    "Patient has Heart Disease": (
        "*Possible Causes:*\n"
        "- High blood pressure and high cholesterol levels\n"
        "- Smoking and excessive alcohol consumption\n"
        "- Lack of physical activity\n"
        "- Poor diet (high in saturated fats and sugars)\n"
        "- Family history of heart disease\n\n"
        "*Precautions:*\n"
        "- Maintain a healthy weight and balanced diet\n"
        "- Engage in regular physical activity (at least 30 minutes daily)\n"
        "- Manage blood pressure and cholesterol levels\n"
        "- Avoid tobacco and limit alcohol consumption\n\n"
        "*Life Risk:*\n"
        "- High if untreated; can lead to heart attack, stroke, or even death."
    ),
    "Patient has No Heart Disease": (
        "- No heart disease detected. Maintain a healthy lifestyle!"
    )
}

kidney_info = {
    "Normal": (
        "- No abnormal findings; kidneys are functioning normally.\n\n"
        "*Precautions:*\n"
        "- Maintain a balanced diet."
    ),
    "Cyst": (
        "*Possible Causes:*\n"
        "- May develop as part of normal aging or due to genetic conditions like Polycystic Kidney Disease (PKD).\n\n"
        "*Precautions:*\n"
        "- Regular monitoring with imaging tests (ultrasound, CT, or MRI).\n"
        "- Maintain proper hydration and blood pressure control.\n\n"
        "*Life Risk:*\n"
        "- Generally low risk."
    ),
    "Stone": (
        "*Possible Causes:*\n"
        "- Formed by crystallization of substances (calcium, oxalate, uric acid) due to dehydration or dietary factors.\n\n"
        "*Precautions:*\n"
        "- Increase water intake (2-3 liters per day).\n"
        "- Adjust diet to reduce salt, animal protein, and oxalate-rich foods.\n\n"
        "*Life Risk:*\n"
        "- Can cause severe pain and complications if untreated."
    ),
    "Tumor": (
        "*Possible Causes:*\n"
        "- Abnormal growth in kidney tissue; risk factors include genetic predisposition, smoking, obesity, and high blood pressure.\n\n"
        "*Precautions:*\n"
        "- Regular screening via imaging tests (ultrasound, CT, or MRI) is essential for high-risk individuals.\n"
        "- Maintain a healthy lifestyle, control weight and blood pressure, and avoid smoking.\n\n"
        "*Life Risk:*\n"
        "- Potentially life-threatening if malignant; early detection and treatment are critical."
    )
}

# ---------------------- SESSION STATE INITIALIZATION ----------------------
if "diabetes_result" not in st.session_state:
    st.session_state.diabetes_result = None
if "heart_result" not in st.session_state:
    st.session_state.heart_result = None
if "kidney_result" not in st.session_state:
    st.session_state.kidney_result = None

# ---------------------- UI: DISEASE SELECTION ----------------------
st.title("Health Risk Prediction")
disease = st.selectbox("Select a disease to predict:", ["Diabetes", "Heart Disease", "Kidney Disease"])

# ---------------------- UI: ALGORITHM SELECTION & INPUTS ----------------------
if disease == "Diabetes":
    st.header("Diabetes Prediction")
    algorithm = "Random Forest"
elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    algorithm = "KNN"
elif disease == "Kidney Disease":
    st.header("Kidney Disease Detection")
    # No algorithm selection for kidney disease

# ---------------------- COLLECT INPUTS ----------------------
if disease in ["Diabetes", "Heart Disease"]:
    st.subheader("Enter Patient Details")
    gender = st.selectbox("Gender", ["Select", "Male", "Female"], index=0)
    age = st.number_input("Age", min_value=1, max_value=100, step=1, format="%d")
    gender_val = 0
    if gender == "Male":
        gender_val = 1

if disease == "Diabetes":
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
    heart_disease_input = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, format="%.1f")
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, step=0.1, format="%.1f")
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, step=1, format="%d")
    smoking_history_current = st.selectbox("Smoking History - Current", [0, 1])
    smoking_history_ever = st.selectbox("Smoking History - Ever", [0, 1])
    smoking_history_former = st.selectbox("Smoking History - Former", [0, 1])
    smoking_history_never = st.selectbox("Smoking History - Never", [0, 1])
    smoking_history_not_current = st.selectbox("Smoking History - Not Current", [0, 1])
    
    input_df = pd.DataFrame([[gender_val, age, hypertension, heart_disease_input, bmi, 
                              HbA1c_level, blood_glucose_level,
                              smoking_history_current, smoking_history_ever, 
                              smoking_history_former, smoking_history_never, 
                              smoking_history_not_current]],
                             columns=["gender", "age", "hypertension", "heart_disease", "bmi", "HbA1c_level", 
                                      "blood_glucose_level", "smoking_history_current", "smoking_history_ever", 
                                      "smoking_history_former", "smoking_history_never", "smoking_history_not_current"])
    if hasattr(diabetes_scaler, "feature_names_in_"):
        input_data = diabetes_scaler.transform(input_df)
    else:
        input_data = diabetes_scaler.transform(input_df.values)

elif disease == "Heart Disease":
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=90, max_value=200, step=1, format="%d")
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, step=1, format="%d")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, step=1, format="%d")
    exang = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, step=0.1, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1, format="%d")
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])
    
    input_df = pd.DataFrame([[age, gender_val, cp, trestbps, chol, fbs, restecg, 
                              thalach, exang, oldpeak, slope, ca, thal]],
                             columns=["age", "sex", "cp", "trestbps", "chol", "fbs", 
                                      "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
    heart_encoded = pd.get_dummies(input_df, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"])
    expected_cols = list(heart_scaler.feature_names_in_)
    for col in expected_cols:
        if col not in heart_encoded.columns:
            heart_encoded[col] = 0
    heart_encoded = heart_encoded[expected_cols]
    input_data = pd.DataFrame(heart_scaler.transform(heart_encoded), columns=expected_cols)

elif disease == "Kidney Disease":
    #st.title("Kidney Disease Detection")
    st.subheader("Upload a CT-SCAN Image")
    uploaded_file = st.file_uploader("Upload a Kidney Image", type=["jpg", "jpeg", "png"])
    
    def preprocess_image(image_bytes):
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error loading image. Please check the file.")
            return None
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_data = preprocess_image(uploaded_file.read())
        if image_data is not None:
            prediction = kidney_model.predict(image_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_labels = ["Cyst", "Normal", "Stone", "Tumor"]
            kidney_result = class_labels[predicted_class]
            st.success(f"Prediction: {kidney_result}")
            st.write("### Additional Information:")
            st.markdown(kidney_info[kidney_result])
            kidney_map = {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}
            st.session_state.kidney_result = kidney_map[kidney_result]
        else:
            st.warning("Please upload a valid image.")
    # Kidney Disease prediction is made immediately via image upload.
    # We do not use the individual Predict button for Kidney Disease.

# ---------------------- INDIVIDUAL DISEASE PREDICTION (DIABETES & HEART) ----------------------
final_result = ""
if st.button("Predict") and disease in ["Diabetes", "Heart Disease"]:
    if gender == "Select":
        st.warning("Please select a valid gender before predicting.")
    else:
        if disease == "Diabetes":
            if HbA1c_level < 6.5 and blood_glucose_level < 126:
                final_result = "Patient is Non-Diabetic"
            elif HbA1c_level >= 6.5 and blood_glucose_level >= 126:
                final_result = "Patient is Diabetic"
            else:
                model_used = rf_model
                prediction = model_used.predict(input_data)
                final_result = "Patient is Diabetic" if int(prediction[0]) == 1 else "Patient is Non-Diabetic"
            st.session_state.diabetes_result = 1 if final_result == "Patient is Diabetic" else 0

        elif disease == "Heart Disease":
            #model_mapping = {"KNN": knn_model, "SVM": svm_model, "Decision Tree": dt_model, "Random Forest": rf_heart_model}
            model_used = knn_model
            if model_used:
                prediction = model_used.predict(input_data)
                final_result = "Patient has Heart Disease" if int(prediction[0]) == 1 else "Patient has No Heart Disease"
                st.session_state.heart_result = 1 if final_result == "Patient has Heart Disease" else 0
            else:
                st.error("Error in selecting model.")
                final_result = ""
        
        if final_result:
            st.success(f"Prediction: {final_result}")
            if disease == "Diabetes":
                st.markdown(diabetes_info.get(final_result, "- No additional information available."))
            elif disease == "Heart Disease":
                st.markdown(heart_info.get(final_result, "- No additional information available."))

# ---------------------- OVERALL HEALTH RISK PREDICTION ----------------------
# This button is displayed only if predictions for all 3 diseases are available.
if (st.session_state.get("diabetes_result") is not None and 
    st.session_state.get("heart_result") is not None and 
    st.session_state.get("kidney_result") is not None):
    if st.button("Predict Overall Health Risk"):
        d = st.session_state.diabetes_result   # 0 or 1
        h = st.session_state.heart_result        # 0 or 1
        k = st.session_state.kidney_result       # 0: Cyst, 1: Normal, 2: Stone, 3: Tumor
        if k == 3 or (d == 1 and h == 1 and k in [0, 2, 3]):
            overall_risk = "High Risk"
            st.success(f"Overall Health Risk Prediction: {overall_risk}")
        elif d == 0 and h == 0 and k == 1:
            overall_risk = "No Risk"
            st.success(f"Overall Health Risk Prediction: {overall_risk}")
        else:
            final_model = load_model("models/final_svm_model.pkl") 
            if final_model is not None: 
               final_input_df = pd.DataFrame([[d, h, k]], columns=["diabetes_result", "heart_result", "kidney_result"]) 
               overall_risk_pred = final_model.predict(final_input_df)[0] 
               st.success(f"Overall Health Risk Prediction: {overall_risk_pred}") 
            else: 
               st.error("Final model not found or failed to load.")
        
        
