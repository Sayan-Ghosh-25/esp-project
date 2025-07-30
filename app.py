import streamlit as st
import pandas as pd
import pickle

# ─── Model Loading ───
model = pickle.load(open('Model.pkl', 'rb'))
train_cols = pickle.load(open("Train_Cols.pkl", "rb"))

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


# ─── User Inputs ───
# Step 1: Industry Selection
Ins = st.selectbox("Industry", ["Select", "Education", "Finance", "Healthcare", "Manufacturing", "Retail", "Technology", "Transportation"], key="industry_select")
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
            sample_enc = pd.get_dummies(sample, drop_first=False)
            sample_enc = sample_enc.reindex(columns=train_cols, fill_value=0)

            # User Salary Prediction
            pred = model.predict(sample_enc)[0]
            st.success(f"💰 Expected Salary (Per Annum): ₹{pred:,.0f}/-")