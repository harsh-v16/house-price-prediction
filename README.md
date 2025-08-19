## 🏡 House Price Prediction

This project predicts house prices using machine learning regression models. The dataset contains housing features such as area, location, rooms, and other attributes. After performing data cleaning and feature engineering, models like Linear Regression and Random Forest were trained and evaluated.

-------------------------

## 📂 Dataset

The dataset includes:
	•	Numerical features (e.g., area, year built, number of rooms)
	•	Categorical features (e.g., location, style)
	•	Target: SalePrice

----------------

## ⚙ Steps Followed
	1.	Data Exploration
	•	Histograms, QQ plots for distribution check
	•	Correlation heatmap to identify important features
	2.	Data Cleaning & Feature Engineering
	•	Handled missing values (mean/median/mode/zero/None based on column type)
	•	Fixed skewness using log transformation
	•	Encoded categorical variables
	3.	Modeling
	•	Linear Regression
	•	XGBoost
	•	Compared using MAE (Mean Absolute Error) and RMSLE (Root Mean Squared Log Error)
	4.	Evaluation
	•	Selected the best model based on lowest error score
	•	Generated predictions on test data

--------------------

## 📊 Results
	•	Best Model: XGBoost
	•	Evaluation Metrics:
	•	MAE: value you got
	•	RMSLE: value you got

----------------------------

##  💻 Tech Stack
	•	Python 🐍
	•	Pandas, NumPy
	•	Matplotlib, Seaborn
	•	Scikit-learn

 ----------

 ## 🚀 How to Run
1.	Clone the repository :
 git clone https://github.com/harsh-v16/house-price-prediction.git
cd house-price-prediction
2.	Install dependencies :
   pip install -r requirements.txt
3.	Run the script :
   python house_rate.py

----------------------------

## 👨‍💻 Author

Harsh Chaudhary
