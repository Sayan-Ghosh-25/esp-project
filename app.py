import os
import requests
import gdown
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ─── Parameters ───
df_columns = ['Industry', 'Job_Title', 'Experience_Level', 'Educational_Level', 'Location']
CSV_URL   = "https://drive.google.com/uc?export=download&id=15iVSngbLNU_Z6zcGl_ZZawG_dDbd-pih"
CSV_FILE  = "Refined Employers Data.csv"

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
    df.dropna(inplace=True)

    np.random.seed(37)
    df['Salary'] = (df['Salary'] * np.random.uniform(0.95, 1.05, size=len(df))).round().astype(int)

    ip = df.drop(columns=['Salary'])
    op = df['Salary']

    ip_train, ip_test, op_train, op_test = train_test_split(ip, op, test_size=0.2, random_state=37)
    ip_train_enc = pd.get_dummies(ip_train,columns=df_columns,drop_first=False)
    return df, ip_train_enc, op_train, ip_test, op_test, ip_train_enc.columns.to_list()

# ─── Train Model (Cached) ───
@st.cache_resource
def train_model(ip_train_enc, op_train):
    rf = RandomForestRegressor(n_estimators=450, max_depth = None, random_state=37, n_jobs=-1)
    rf.fit(ip_train_enc, op_train)
    return rf

# ─── Main UI ───
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 *Salary Prediction System*")
st.header("Enter Employee Details")

# ─── Reset Helper ───
def clear_all():
    keys = ["industry_select", "job_select", "exp_select", "edu_select", "location_select", "previous_industry"]
    if all(st.session_state.get(k, "Select") == "Select" for k in keys):
        st.session_state.show_clear_error = True
    else:
        for k in keys:
            st.session_state[k] = "Select"
        st.session_state.show_clear_error = False

if st.session_state.get("show_clear_error", False):
    st.error("⚠️ Nothing To Clear!")
    st.session_state.show_clear_error = False

# ─── Global Loading ───
df, ip_train_enc, op_train, ip_test, op_test, MODEL_COLS = load_data()
model = train_model(ip_train_enc, op_train)

# ─── User Inputs ───
# Step 1: Industry Selection
Ins = st.selectbox("Industry", ["Select"] + sorted(df['Industry'].unique()), key="industry_select")
if "previous_industry" not in st.session_state or st.session_state.previous_industry != Ins:
    st.session_state.job_select = "Select"
st.session_state.previous_industry = Ins

# Step 2: Job Titles Based on Industry
job_options = ["Select"]

if Ins == "Finance":
    job_options += ["HR", "Analyst", "Manager", "Executive"]
elif Ins == "Education":
    job_options += ["Clerk", "Teacher", "Department Head", "Institute Head"]
elif Ins in ["Technology", "Transportation", "Manufacturing"]:
    job_options += ["Intern", "Engineer", "HR", "Manager", "Executive"]
elif Ins == "Healthcare":
    job_options += ["Staff", "Doctor", "Management Head", "Executive"]
elif Ins == "Retail":
    job_options += ["Intern", "Analyst", "Manager", "Executive"]
else:
    job_options += sorted(df[df["Industry"] == Ins]["Job_Title"].unique())

Job = st.selectbox("Job Title", job_options, key="job_select")

# Step 3: Experience Level Based on Job
exp_options = ["Select"]

if Job == "Intern":
    exp_options += ["Entry-Level"]
elif Job in ["Analyst", "Staff", "Clerk"]:
    exp_options += ["Entry-Level", "Mid-Level"]
elif Job == "HR":
    exp_options += ["Mid-Level"]
elif Job in ["Engineer", "Teacher", "Doctor"]:
    exp_options += ["Entry-Level", "Mid-Level", "Senior-Level"]
elif Job in ["Manager", "Department Head", "Management Head"]:
    exp_options += ["Mid-Level", "Senior-Level"]
elif Job in ["Executive", "Institute Head"]:
    exp_options += ["Senior-Level"]

Exp = st.selectbox("Experience Level", exp_options, key="exp_select")

# Step 4: Education Level Based on Experience
edu_options = ["Select"]

if Exp in ["Entry-Level", "Mid-Level"]:
    edu_options += ["Bachelor", "Master"]
elif Exp == "Senior-Level":
    edu_options += ["Master", "PhD"]

Edu = st.selectbox("Educational Qualification", edu_options, key="edu_select")

# Step 5: Location
loc_options = ["Select"]

if Exp != "Select":
    loc_options += ["America", "Australia", "Dubai", "England", "India"]

Loc = st.selectbox("Location", loc_options, key="location_select")

# ─── Buttons Section ───
col1, col2 = st.columns([1,7])
with col1:
    # ─── Clear Selections ───
    st.button("Clear All", key="clear_choices", on_click=clear_all)

with col2:
    # ─── Model Prediction ───
    if st.button("Check Salary", key="check_salary"):
        if "Select" in [Ins, Job, Edu, Loc]:
            st.error("⚠️ Please Fill All The Fields!")
        else:
            sample = pd.DataFrame([{
                'Industry': Ins.strip().title(),
                'Job_Title': Job.strip().title(),
                'Experience_Level': Exp.strip().title(),
                'Educational_Level': Edu.strip().title(),
                'Location': Loc.strip().title(),
            }])

            # User Input Encoding
            sample_enc = pd.get_dummies(sample, columns=df_columns, drop_first=False)
            sample_enc = sample_enc.reindex(columns=MODEL_COLS, fill_value=0)

            # User Salary Prediction
            pred = model.predict(sample_enc)[0]
            st.success(f"💰 Expected Salary (Per Annum): ₹{pred:,.0f}/-")