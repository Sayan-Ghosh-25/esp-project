import os
import requests
import gdown
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ─── Parameters ───
df_columns = ['Gender','Department','Job_Title','Education_Level','Location']
CSV_URL   = "https://drive.google.com/uc?export=download&id=1ghlU4g5ISe1_Q5FDF7N9g_POLVszRJBg"
CSV_FILE  = "Employers_Data.csv"

# ─── Download Helper ───
def download_if_missing(url: str, filename: str):
    if not os.path.exists(filename):
        try:
            gdown.download(url, filename, quiet=False)
        except Exception:
            with st.spinner(f"Downloading {filename}..."):
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

# ─── Load, Clean & Encode Train (Cached) ───
@st.cache_data
def load_data():
    download_if_missing(CSV_URL, CSV_FILE)

    df = pd.read_csv(CSV_FILE)
    df.drop(columns=['Name','Employee_ID'], inplace=True)
    df.dropna(inplace=True)

    df.replace({'Location':{
        'Austin':'India','Seattle':'Australia',
        'Chicago':'America','New York':'England',
        'San Francisco':'Dubai'
    }}, inplace=True)

    ip = df.drop(columns=['Salary'])
    op = df['Salary']

    ip_train, ip_test, op_train, op_test = train_test_split(ip, op, test_size=0.2, random_state=42)

    ip_train_enc = pd.get_dummies(ip_train,columns=df_columns,drop_first=False)

    return df, ip_train_enc, op_train, ip_test, op_test, ip_train_enc.columns.to_list()

# ─── Train Model (Cached) ───
@st.cache_resource
def train_model(ip_train_enc, op_train):
    rf = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
    rf.fit(ip_train_enc, op_train)
    return rf

# ─── Main UI ───
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 *Salary Prediction System*")
st.header("Enter Employee Details")

# Global Loading
df, X_train_enc, y_train, X_test, y_test, MODEL_COLS = load_data()
model = train_model(X_train_enc, y_train)

# User Inputs
Age = st.number_input("Age", 18, 70, 25, key="age_select")
Gender = st.selectbox("Gender", ["Select"] + sorted(df['Gender'].unique()), key="gender_select")
Dept   = st.selectbox("Department", ["Select"] + sorted(df['Department'].unique()), key="department_select")
Job    = st.selectbox("Job Title", ["Select"] + sorted(df['Job_Title'].unique()), key="job_title_select")
Exp    = st.number_input("Experience Years", 0, 50, 2, key="experience_years_select")
Edu    = st.selectbox("Education Level", ["Select"] + sorted(df['Education_Level'].unique()), key="education_level_select")
Loc    = st.selectbox("Location", ["Select"] + sorted(df['Location'].unique()), key="location_select")

# Model Prediction
if st.button("Check Salary", key="check_salary"):
    if "Select" in [Gender,Dept,Job,Edu,Loc]:
        st.error("⚠️ Please Fill All The Fields!")
    else:
        sample = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender.strip().title(),
            'Department': Dept.strip().title(),
            'Job_Title': Job.strip().title(),
            'Experience_Years': Exp,
            'Education_Level': Edu.strip().title(),
            'Location': Loc.strip().title(),
        }])

        # User Input Encoding
        sample_enc = pd.get_dummies(sample, columns=df_columns, drop_first=False)
        sample_enc = sample_enc.reindex(columns=MODEL_COLS, fill_value=0)

        # User Salary Prediction
        pred = model.predict(sample_enc)[0]
        st.success(f"💰 Expected Salary: ₹{pred:,.0f}")