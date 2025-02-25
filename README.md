﻿# Sentiment Analysis Web Application

## 📌 Overview
This project is a **Sentiment Analysis Web Application** that classifies text as **Positive, Negative, or Neutral** using **VADER** and a retrainable **Naive Bayes model**. The app allows user feedback to improve accuracy over time.

## 🚀 Features
- **Real-time Sentiment Analysis** (Flask + VADER/NLP model)
- **User Feedback Mechanism** (Users can confirm or correct results)
- **Retraining System** (Model updates based on feedback)
- **Responsive UI** (Built with React.js)
- **Clear Button** (Resets the input field easily)

## 🛠 Tech Stack
- **Frontend:** React.js  
- **Backend:** Flask (Python)  
- **NLP Model:** VADER, Naïve Bayes  
- **Database:** CSV (for user feedback storage)  

## 🏗 How to Run
1. **Start Backend**  
```sh
python api.py
```
2. **Start Frontend**  
```sh
npm start
```
3. **Retrain Model (After collecting feedback)**  
```sh
curl http://127.0.0.1:5000/retrain
```

## 📌 Future Improvements
- Implement **deep learning models (BERT)** for better classification.
- Add **multilingual support** for non-English texts.
- Optimize **feedback processing** for better training data.

---
Created by **Bekhzod Allaev & Akbarali Nabiev**
