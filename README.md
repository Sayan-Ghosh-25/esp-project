# 💼 Salary Predictor App

Welcome to your **AI-powered Employee Salary Estimator**!  
This web app predicts the expected salary of an employee based on professional attributes like age, experience, job title, location, education level, and more.

🔗 **Live App**: [Launch Salary Predictor](https://salary-predictor-sg25.streamlit.app)

---

## 📸 Preview

### 🧾 Input Form
![Input Form](https://github.com/Sayan-Ghosh-25/esp-project/blob/master/assets/Preview-1.png?raw=true)

### 📈 Predicted Salary Output
![Prediction Output](https://github.com/Sayan-Ghosh-25/esp-project/blob/master/assets/Preview-2.png?raw=true)

---

## 🚀 Features

- 📋 Fill in employee details using clean, interactive dropdowns and number fields.
- 🔍 Predict salary in real-time based on a **trained machine learning model**.
- 💡 Input values are automatically standardized for accuracy (e.g. title-casing, space trimming).
- ⚙️ Dataset automatically downloaded from **Google Drive** if not found locally.
- 🔐 Robust model feature alignment prevents shape mismatch or incorrect predictions.
- ⚡ Fast performance using Streamlit’s built-in caching (`st.cache_data` & `st.cache_resource`).
- 🧑‍💼 Built for HR, recruiters, analysts, and developers to test salary scenarios.

---

## 🧠 How It Works

- Built using **Streamlit** for the front-end UI.
- Uses a **Random Forest Regressor** from `scikit-learn`, trained on a cleaned employee dataset.
- Categorical features are fully one-hot encoded (no `drop_first`), ensuring full feature capture.
- The prediction is made based on the trained model using real-time user input.

---

## 📦 Tech Stack

- `streamlit` – UI & deployment
- `pandas` – Data manipulation
- `numpy` – Numerical analysis
- `scikit-learn` – Model training and prediction
- `gdown`, `requests` – Fetch dataset from Google Drive

---

## 🛠️ Project Structure

```
Salary-Predictor/
│
├── .streamlit/
│   └── config.toml        # Streamlit Cloud Settings
├── assets/
│   ├── Preview-1.png # Input form screenshot
│   └── Preview-2.png # Prediction output screenshot
├── app.py # Streamlit main app
├── requirements.txt # Python Dependencies
└── README.md # You're Reading It!
```

---

## 🧾 License

-  This project is created for educational and demo purposes.
- All employee data is fictional and used only for machine learning practice.

---

## 🙋‍♂️ Author

**Sayan Ghosh**  
Feel Free To Connect Me On [LinkedIn](https://www.linkedin.com/in/sayan-ghosh25) or Contribute To This Project 😎

---

## ⭐ Acknowledgement

- Thanks to the Streamlit and Scikit-learn communities for providing powerful tools.
- Inspired by practical use-cases in HR analytics and data-driven compensation systems.