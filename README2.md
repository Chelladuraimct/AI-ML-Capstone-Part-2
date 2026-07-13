# Part 2 – Supervised Machine Learning

## Objective

After cleaning and preprocessing the dataset in Part 1, supervised machine learning models were developed for both regression and classification tasks.

The objective was to build predictive models, evaluate their performance, compare different algorithms, and understand how preprocessing decisions affect model performance.

---

# Data Preparation

The cleaned dataset generated in Part 1 (`cleaned_data.csv`) was loaded into a pandas DataFrame.

The dataset contained both numerical and categorical features.

The feature matrix (`X`) contained all predictor variables, while two target variables were defined:

- **Regression Target (`y_reg`)**: A continuous numerical column selected from the dataset.
- **Classification Target (`y_clf`)**: A binary target created by comparing the regression target with its median value.

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

This converted the regression problem into a binary classification problem suitable for Logistic Regression.

---

# Feature Encoding

Categorical variables cannot be directly used by machine learning algorithms.

Two encoding techniques were considered:

### Label Encoding

Label Encoding assigns integer values to categories while preserving natural order.

Example:

```
Low → 0
Medium → 1
High → 2
```

This technique is appropriate only when categories have an inherent ordering.

---

### One-Hot Encoding

For categorical variables without any natural ordering, One-Hot Encoding was applied using:

```python
pd.get_dummies(drop_first=True)
```

This creates binary indicator variables for each category.

Dropping the first dummy variable avoids multicollinearity (the dummy variable trap).

Unlike Label Encoding, One-Hot Encoding prevents the model from incorrectly assuming ordinal relationships between categories.

---

# Train-Test Split

The dataset was divided into training and testing subsets using:

```python
train_test_split(
    test_size=0.20,
    random_state=42
)
```

- 80% Training Data
- 20% Testing Data

The same random seed ensures reproducibility.

---

# Feature Scaling

Standardization was performed using:

```python
StandardScaler()
```

The scaler was fitted **only on the training dataset** and then applied to both training and testing data.

This avoids **data leakage**, where information from the test set unintentionally influences the training process.

If the scaler were fitted on the entire dataset, the test-set statistics (mean and standard deviation) would leak into training, leading to overly optimistic evaluation results.

---

# Regression Model – Linear Regression

A Linear Regression model was trained using the standardized training features.

Performance was evaluated using:

- Mean Squared Error (MSE)
- R² Score

The model coefficients were extracted and matched with their corresponding feature names.

The three features with the largest absolute coefficient values were identified as the strongest predictors.

### Interpretation

A positive coefficient indicates that increasing the standardized feature tends to increase the predicted target value.

A negative coefficient indicates that increasing the standardized feature tends to decrease the predicted target value.

Features with larger absolute coefficient values have a stronger influence on the prediction.

---

# Ridge Regression

To reduce overfitting caused by multicollinearity, Ridge Regression was trained using:

```python
Ridge(alpha=1.0)
```

Performance metrics (MSE and R²) were compared against ordinary Linear Regression.

### Role of Alpha

The alpha parameter controls the amount of L2 regularization.

- Small alpha → behaves similarly to ordinary least squares.
- Large alpha → stronger regularization, shrinking coefficients toward zero.

Ridge Regression often produces more stable coefficient estimates when correlated predictors are present.

---

# Binary Classification – Logistic Regression

A Logistic Regression classifier was trained using:

```python
LogisticRegression(max_iter=1000)
```

If class imbalance was observed, appropriate balancing techniques such as `class_weight='balanced'` or SMOTE were applied to the training data only.

The classifier generated both class predictions and probability estimates.

---

# Classification Metrics

Model performance was evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

### Precision

\[
Precision = \frac{TP}{TP + FP}
\]

Precision measures the proportion of predicted positive cases that are actually positive.

---

### Recall

\[
Recall = \frac{TP}{TP + FN}
\]

Recall measures the proportion of actual positive cases correctly identified by the model.

---

### F1 Score

The F1 Score is the harmonic mean of Precision and Recall.

It provides a balanced measure when both false positives and false negatives are important.

---

# ROC Curve

The Receiver Operating Characteristic (ROC) Curve illustrates the trade-off between:

- True Positive Rate
- False Positive Rate

The curve was generated using:

```python
roc_curve()
```

---

# Area Under Curve (AUC)

The ROC-AUC score summarizes classifier performance across all classification thresholds.

Interpretation:

- AUC = 1.0 → Perfect classifier
- AUC = 0.9 → Excellent
- AUC = 0.8 → Good
- AUC = 0.5 → Random guessing

Higher AUC values indicate better separation between positive and negative classes.

---

# Decision Threshold Analysis

Rather than always using the default probability threshold of **0.50**, thresholds from:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

were evaluated.

For each threshold, the following metrics were computed:

- Precision
- Recall
- F1 Score

The threshold that produced the highest F1 Score was identified.

### Interpretation

Lower thresholds generally increase Recall but reduce Precision.

Higher thresholds increase Precision while decreasing Recall.

The optimal threshold depends on the application's tolerance for false positives and false negatives.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained using:

```python
C = 0.01
```

The value of **C** controls inverse regularization strength.

- Large C → weaker regularization.
- Small C → stronger regularization.

Performance metrics (Precision, Recall, and AUC) were compared against the baseline model.

Reducing C simplifies the model but may reduce predictive performance if excessive regularization removes useful information.

---

# Bootstrap Confidence Interval

Bootstrap sampling was performed using **500 resamples**.

For each bootstrap sample:

- Test observations were sampled with replacement.
- AUC was calculated for both Logistic Regression models.
- The difference in AUC values was recorded.

The following statistics were computed:

- Mean AUC Difference
- 2.5th Percentile
- 97.5th Percentile

These values formed the **95% Confidence Interval**.

### Interpretation

If the confidence interval excludes zero, the performance difference between models is statistically reliable.

If zero lies inside the interval, the observed difference may simply be due to sampling variation.

---

# Part 2 Summary

This section demonstrated the complete supervised machine learning workflow.

Major activities included:

- Feature Encoding
- Data Scaling
- Train-Test Splitting
- Linear Regression
- Ridge Regression
- Logistic Regression
- ROC Analysis
- Threshold Optimization
- Regularization Comparison
- Bootstrap Confidence Interval Estimation

The resulting models provide the foundation for the advanced ensemble methods implemented in Part 3.
