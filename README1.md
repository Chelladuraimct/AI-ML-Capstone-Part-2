# Applied AI & ML Essentials Capstone Project

## Student Information

**Name:** R. Chelladurai

**Course:** Applied AI & ML Essentials

**Project:** Machine Learning Capstone

---

# Project Overview

This capstone project demonstrates the complete Machine Learning workflow from raw data preprocessing to advanced predictive modeling using Python and Scikit-learn.

The project is divided into three major parts:

- Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis
- Part 2 – Supervised Machine Learning Models
- Part 3 – Advanced Machine Learning, Ensemble Models, Hyperparameter Tuning and Model Deployment

The primary objective is to transform a raw dataset into a clean, analysis-ready dataset, build multiple predictive models, evaluate them using various performance metrics, and finally recommend the most robust model for deployment.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

# Dataset Description

The original dataset contained both numerical and categorical variables with missing values, duplicate records, inconsistent data types and potential outliers.

Before training machine learning models, several preprocessing steps were performed to improve data quality and reliability.

---

# Repository Structure

```
Capstone_Project/
│
├── Capstone_Part1.ipynb
├── cleaned_data.csv
├── best_model.pkl
├── README.md
└── requirements.txt
```

---

# Part 1 – Data Cleaning and Exploratory Data Analysis

## Loading Dataset

The dataset was loaded using

```python
pd.read_csv()
```

The following information was inspected:

- First five rows
- Dataset shape
- Column names
- Data types

This provided an initial understanding of the dataset structure.

---

# Missing Value Analysis

Missing values were identified using

```python
df.isnull().sum()
```

The percentage of missing values for every column was calculated using

```python
(df.isnull().sum()/len(df))*100
```

Columns containing more than 20% missing values were identified separately.

For columns having fewer than 20% missing values, missing numerical values were replaced using the column median.

---

# Why Median Instead of Mean?

The median is resistant to extreme values and skewed distributions.

The mean is heavily influenced by very large or very small observations.

Since many real-world datasets contain skewed distributions and outliers, median imputation preserves the central tendency more accurately.

---

# Duplicate Detection

Duplicate observations were identified using

```python
df.duplicated().sum()
```

All duplicate rows were removed using

```python
df.drop_duplicates()
```

After removing duplicates, missing-value percentages were recalculated to ensure consistency.

---

# Data Type Conversion

Several columns were converted into more appropriate data types.

Examples include

- Object → Numeric
- Object → Category

Numeric conversion was performed using

```python
pd.to_numeric()
```

Categorical conversion was performed using

```python
astype("category")
```

Memory usage before and after conversion was compared.

The optimized dataset consumed less memory while maintaining identical information.

---

# Descriptive Statistics

Descriptive statistics were generated using

```python
df.describe()
```

The following measures were analyzed:

- Mean
- Median
- Standard Deviation
- Minimum
- Maximum
- Quartiles

These statistics provided a summary of the overall distribution of each numerical feature.

---

# Skewness Analysis

Skewness was calculated for every numerical feature.

The feature having the largest absolute skewness was identified.

Interpretation:

Positive skew indicates that a small number of very large observations pull the distribution toward the right.

Negative skew indicates that unusually small observations pull the distribution toward the left.

Highly skewed variables generally benefit from median imputation because the mean becomes biased toward extreme values.

---

# Outlier Detection Using IQR

Outliers were identified using the Interquartile Range (IQR) method.

The following statistics were computed:

- First Quartile (Q1)
- Third Quartile (Q3)
- Interquartile Range (IQR)

Outliers were defined as observations lying outside

Lower Bound

Q1 − 1.5 × IQR

Upper Bound

Q3 + 1.5 × IQR

Outliers were documented rather than removed because they may contain useful business information.

---

# Data Visualization

The following visualizations were created.

## 1. Line Plot

Purpose:

Visualize trends in a numerical feature.

Interpretation:

The line plot illustrates the variation of observations across the dataset and helps identify unusual spikes or sudden changes.

---

## 2. Bar Chart

Purpose:

Compare average numerical values across different categories.

Interpretation:

The chart highlights categories having relatively higher or lower average values.

---

## 3. Histogram

Purpose:

Understand the distribution of the most skewed numerical feature.

Interpretation:

The histogram reveals whether the distribution is symmetric, positively skewed or negatively skewed.

---

## 4. Scatter Plot

Purpose:

Investigate relationships between two numerical variables.

Interpretation:

The scatter plot demonstrates the strength and direction of correlation between the selected variables.

---

## 5. Box Plot

Purpose:

Compare distributions across categories while identifying potential outliers.

Interpretation:

Differences in medians and spread across categories indicate possible variations between groups.

---

# Correlation Heatmap

Pearson correlation coefficients were computed using

```python
df.corr()
```

The heatmap visualized relationships among all numerical variables.

The pair having the highest absolute correlation was identified.

High correlation does not necessarily imply causation.

A third variable may influence both variables simultaneously, resulting in a strong observed relationship.

---

# Mean vs Median Imputation Comparison

The two most skewed numerical variables were analyzed separately.

For each variable:

- Mean was computed
- Median was computed

Because skewed distributions shift the mean toward extreme observations, median was selected for imputation.

After imputation, missing values were verified using

```python
df.isnull().sum()
```

No missing values remained.

---

# Spearman Rank Correlation

Spearman correlation was calculated using

```python
df.corr(method="spearman")
```

Differences between Pearson and Spearman correlation coefficients were analyzed.

Pairs exhibiting higher Spearman correlation than Pearson correlation indicate monotonic but non-linear relationships.

For feature selection, Pearson correlation was primarily used for approximately linear relationships, while Spearman correlation provided additional insight into monotonic associations.

---

# Grouped Aggregation

Grouped statistics were calculated using

```python
df.groupby().agg(["mean","std","count"])
```

The following were identified:

- Group with highest mean
- Group with highest standard deviation

High within-group standard deviation suggests that the categorical variable alone may not be sufficient for accurate prediction.

The ratio between the highest and lowest group means was also computed to estimate the predictive strength of the categorical feature.

---

# Clean Dataset

The cleaned dataset was exported as

```
cleaned_data.csv
```

This dataset serves as the input for Part 2 and Part 3.

---