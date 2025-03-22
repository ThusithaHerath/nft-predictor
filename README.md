# NFT - Predictor 

# 🧠 NFT Sales Prediction Model

This project is an end-to-end machine learning application designed to **predict the success of NFT collections** based on various market indicators and public sentiment analysis. Built with **Flask (Python)** for the backend and **React + Vite + TypeScript + Tailwind CSS** for the frontend, it provides actionable insights for both **new** and **existing** NFT projects.

## 🚀 Features

- 📊 Predict success probability for NFT collections
- 🌍 Recommend best geographical markets
- 💡 Suggest marketing strategies
- 📈 Analyze historical data for existing collections
- 🔍 Sentiment analysis using public data (Twitter, Discord)
- 📁 Dataset preprocessing: cleaning, scaling, feature engineering
- 🔬 ML models using **XGBoost**, **Scikit-learn**, etc.
- 🧪 API testing with **Postman**
- ☁️ Hosted with **Firebase** (frontend) and **Render** (backend)

---

## 🛠️ Tech Stack

### 🔧 Backend (Flask)
- Python 3.x
- Flask + Flask-CORS
- Pandas, NumPy, Scikit-learn, XGBoost
- Render (Deployment)

### 💻 Frontend (React)
- Vite + React + TypeScript
- Tailwind CSS
- Firebase Hosting

### 🗃️ Dataset
Includes features such as:
- Collection name, category, volume, market cap, average price, number of owners
- Sentiment scores
- Roadmap strength and social media engagement (for new projects)

---

## 📈 Model Workflow

1. User inputs NFT collection details
2. App checks if it's a new or existing project
3. Preprocess data (scaling, balancing, encoding)
4. Run model inference using trained ML models
5. Display success prediction, recommendations, and insights

---

## 🔍 How to Run Locally

### Backend (Flask API)

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
