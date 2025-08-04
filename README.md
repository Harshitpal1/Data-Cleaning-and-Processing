# Data-Cleaning-and-Processing
# AI & ML Internship Task 1: Data Cleaning & Preprocessing

This repository contains the solution for Task 1 of the AI & ML Internship with Elevate Labs and the Ministry of MSME. The objective of this task is to clean and preprocess a dataset to make it suitable for machine learning algorithms.

## Objective

The primary goal is to learn and apply various data cleaning and preparation techniques to a raw dataset. This involves handling missing values, encoding categorical data, scaling numerical features, and managing outliers.

## Tools Used

* Python.
* Pandas.
* NumPy
* Matplotlib 

## Steps Followed

The entire process is broken down into the following steps as per the task guidelines:

1.  *Import Dataset and Explore*: The Titanic dataset was loaded into a Pandas DataFrame. Basic information, such as data types and the presence of null values, was examined.
2.  Handle Missing Values: Missing data in the 'Age' column was filled using the median, while the 'Embarked' column's missing values were filled with the mode. The 'Cabin' column was dropped due to a high number of missing entries.
3.  Categorical Features: Categorical columns ('Sex', 'Embarked') were converted into numerical format using one-hot encoding to make them compatible with ML models.
4.  Normalize Numerical Features**: Numerical features ('Age', 'Fare', etc.) were standardized using StandardScaler to bring them to a common scale.
5.  Visualize and Remove Outliers Boxplots were used to identify outliers in the 'Age' and 'Fare' columns. These outliers were then removed using the Interquartile Range (IQR) method to improve model robustness.

## Key Learnings

This task provided hands-on experience with fundamental data preprocessing techniques, including:
*Data cleaning and exploration.
* Handling null values using different strategies
* Encoding categorical data model consumption.
* The importance and application of feature scaling

## Interview Questions & Answers

1.What are the different types of missing data?
   - Missing Completely at Random (MCAR):The absence of data is not related to any other variable or the missing data itself.
   - Missing at Random (MAR): The absence of data is related to other observed variables but not the missing data itself.
   - Missing Not at Random (MNAR):The absence of data is related to the value of the missing data itself.

2. How do you handle categorical variables?
   - You can convert them into numbers using techniques like One-Hot Encoding (creating binary columns for each category) or Label Encoding (assigning a unique integer to each category).

3. What is the difference between normalization and standardization?**
   - Normalization scales data to a fixed range, typically 0 to 1.
   - Standardization transforms data to have a mean of 0 and a standard deviation of 1.

4. How do you detect outliers?
   - Visually: Using boxplots, scatter plots, or histograms.
   - Statistically:Using methods like the Z-score or the Interquartile Range (IQR).

5.Why is preprocessing important in ML
   - It is essential for cleaning raw, noisy data to improve the accuracy, efficiency, and reliability of machine learning models. Real-world data is rarely perfect.

What is one-hot encoding vs label encoding?
   - One-Hot Encoding is used for nominal categorical data where no ordinal relationship exists. It creates a new binary feature for each category.
   - Label Encodingis used for ordinal categorical data, assigning integers in a specific order. Using it for nominal data can mislead the model into assuming an order.

7. How do you handle data imbalance?
   - Resampling: Oversampling the minority class (e.g., SMOTE) or undersampling the majority class.
   - Using appropriate metrics:Focusing on Precision, Recall, or F1-score instead of just accuracy.
   - Algorithmic approaches: Using models that handle imbalance well or adjusting class weights.

8. Can preprocessing affect model accuracy?
   - Yes, significantly. Effective preprocessing can dramatically improve model accuracywhile incorrect preprocessing can harm it.

## How to Run the Code

1.  Clone this repository.
2.  Make sure you have the required libraries installed.
3.  Place the titanic.csv dataset in the same directory.
4.  Run the Python script data_preprocessing.py.

