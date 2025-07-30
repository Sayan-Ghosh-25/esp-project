# 💼 Salary Predictor App

Welcome to the **AI-Powered Salary Prediction System**!  
This intelligent web app predicts the **annual salary** of an employee based on their job profile, industry, experience level, education, and location.

🔗 **Live App**: [Launch Salary Predictor](https://salary-predictor-sg25.streamlit.app)

---

## 📸 Preview

### 🧾 Input Form
![Input Form](https://github.com/Sayan-Ghosh-25/esp-project/blob/master/assets/Preview-1.png?raw=true)

### 📈 Predicted Salary Output
![Prediction Output](https://github.com/Sayan-Ghosh-25/esp-project/blob/master/assets/Preview-2.png?raw=true)

---

## 🚀 Features

- 🎛️ Smart dropdown system that dynamically adjusts based on previous selections (e.g., Industry → Job → Experience).
- 🧠 Salary predicted in real-time using a **trained ML model** (Random Forest Regressor).
- 💡 Intelligent form cleanup with reset handling and empty validation.
- 🔐 Automatic feature alignment via `one-hot encoding` to match training-time schema.
- 📂 Compatible with `.pkl` model loading for fast offline predictions.
- ✅ Clean UI built using **Streamlit** with a centered layout.

---

## ⚙️ How It Works

- The app is built entirely with **Streamlit** for the frontend.
- A **Random Forest Regressor** is trained on cleaned employee data.
- User input is captured through dropdowns and encoded using `pd.get_dummies()`.
- The encoded data is reindexed to match training columns before prediction.
- Predictions are displayed instantly with proper formatting.

---

## 🧠 Tech Stack

- `streamlit` – Web interface
- `pandas` – Data handling
- `scikit-learn` – Machine learning
- `pickle` – Model loading
- `python` – Backend logic

---

## 📁 Project Structure

```
Salary-Predictor/
│
├── assets/
│ ├── Preview-1.png # Screenshot of Input Form
│ └── Preview-2.png # Screenshot of Output
├── Model.pkl # Trained Random Forest Model
├── Train_Cols.pkl # Column List for Reindexing
├── app.py # Main Streamlit App
├── requirements.txt # Python Dependencies
├── FESP.ipynb # Model Training Code
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