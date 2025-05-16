# 🚗 Car Price Prediction using Machine Learning

This project helps users predict the **resale price of a car** based on multiple features such as brand, fuel type, transmission, usage, and more. It leverages machine learning models and presents the insights through a visually appealing, interactive **Streamlit web application**.

## 🔗 **Live App 👉 [Car Price Prediction - Streamlit App](#)**
*(Click the link to predict resale prices and explore data trends.)*

---

## 📌 Problem Statement

Estimating the resale value of a car can be complex due to the wide range of influencing factors. This project aims to **predict the selling price** of used cars using a trained ML model based on real-world data. It also includes **exploratory data analysis (EDA)** to uncover patterns in automobile pricing.

---

## 📊 Dataset Overview

📂 **Source:** Public Dataset (`car data.csv`)

### Features Used:

| Feature        | Description                          |
| -------------- | ------------------------------------ |
| Car\_Name      | Model name of the car                |
| Year           | Year of manufacture                  |
| Present\_Price | Current ex-showroom price (in Lakhs) |
| Selling\_Price | Price at which the car is being sold |
| Kms\_Driven    | Total distance driven                |
| Fuel\_Type     | Petrol, Diesel, or CNG               |
| Seller\_Type   | Dealer or Individual                 |
| Transmission   | Manual or Automatic                  |
| Owner          | Number of previous owners            |

Derived Feature:
🔤 **Brand** – Extracted from `Car_Name`

---

## 🧪 Exploratory Data Analysis (EDA)

✔️ Cleaned and encoded categorical data
✔️ Extracted car brand from model name
✔️ Visualized key relationships in pricing
✔️ Explored influence of fuel type, transmission, and ownership

### 📊 Visualizations Included:

* 🔍 Histogram for Numerical Features
* 📉 Selling vs Present Price Distributions
* 🚘 Top 20 Car Models by Average Selling Price
* 📆 Price Trends by Year
* 🌡️ Correlation Heatmap between features

---

## 🤖 Machine Learning Approach

### 🧠 Models Used:

* Random Forest Regressor (final deployed model)

### 🔧 Preprocessing:

* Label encoding for categorical features
* Feature selection based on correlation
* Model tuning using grid search

---

## 🌐 Streamlit Web App

The app has 3 main sections:

### 🏠 Home

* Introduction
* Skills used
* Domain overview
* Problem and solution summary

### 📈 Get Prediction

* User inputs car details
* Predicts resale price using trained ML model
* Displays result with intuitive UI

### 📊 Explore Data

* Interactive EDA visualizations
* Sidebar filter for selecting topics
* Matplotlib, Seaborn, and Plotly charts

---

## 🛠 Technologies Used

| Tool             | Purpose                     |
| ---------------- | --------------------------- |
| Python 🐍        | Core scripting and logic    |
| Pandas           | Data manipulation           |
| Matplotlib       | Static visualizations       |
| Seaborn          | Statistical plotting        |
| Plotly           | Interactive charts          |
| Streamlit        | Web app deployment          |
| Scikit-learn     | Model building & evaluation |
| Jupyter Notebook | EDA and model training      |

---

## 📈 Sample Visuals

* 📊 Bar Chart: Top Car Models by Selling Price
* 📉 Histogram: Price distribution and mileage
* 🔥 Heatmap: Feature correlations
* ⚙️ Input form and real-time prediction result

---

## 💡 Key Insights

* 🚘 **Brand and year** of manufacture significantly affect price
* 🔁 **Automatic cars** tend to be priced higher
* ⛽ **Fuel type** impacts resale value, with Petrol dominating
* 🧍 Individual sellers offer lower prices than dealers
* 🧭 Lower ownership count increases resale price

---

## 📜 License

This project uses publicly available data and is intended for **educational purposes** only. Refer to the dataset source for licensing information.

---

## 🤝 Connect With Me

💼 **LinkedIn** – [Annie Shilpha](linkedin.com/in/annieshilpha)
📧 **Email** – [shilpha127@gmail.com](mailto:shilpha127@gmail.com)
💻 **GitHub** – [GitHubProfile](https://github.com/Annieshilpha07)

---
