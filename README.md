# ClassifierComparison
 # UC Berkeley AI/ML Course - Professional Certificate ML/AI - Practical Application Assignment 17.1: Comparing Classifiers

**Comparing Classifiers for Bank Marketing Data**

* [Business Understanding](#Business-Understanding)
* [Data Understanding](#Data-Understanding)
* [Data Preparation](#Data-Preparation)
* [Baseline Model Comparison](#Baseline-Model-Comparison)
* [Model Comparisons](#Model-Comparisons)
* [Improving the Model](Improving-the-Model)
* [Next Steps & Recommendations](#Next-Steps-&-Recommendations)

## Business Understanding

The objective of this work is to predict the customers that will subscribe to a long-term bank deposit from the past marketing campaigns. This project processes a dataset of a Portuguese bank, categorizing customer responses with machine learning models. By recognizing factors that influence customer choices, the bank will be able to maximize future marketing campaigns and increase success rates.

Customer demographics, past experience with the bank, and economic indicators make up the dataset. By identifying patterns in the data, targeting effectiveness is improved and failed calls reduced..

## Data Understanding

To know more about the dataset, I carried out an Exploratory Data Analysis, by using various visualizations and statistical summaries: 

### Countplot of Subscription Outcomes
The countplot was employed to plot the proportion of customers who accepted (yes) versus those who declined (no) the offer. 

![image](https://github.com/user-attachments/assets/e9e52efd-d0cb-4a1e-ab15-0368ccf269f2)

The plot readily indicates that the data is highly imbalanced and that the majority of customers declined the offer. The imbalance needs to be addressed in modeling.

### Scatterplot Analysis
Scatterplots were used to examine correlations between attributes such as age vs. call duration to determine if monetary factors play an important role in customer decision-making.

![image](https://github.com/user-attachments/assets/f26b6fc7-4ddf-431a-b070-4904ea428e94)

The plot revealed that calls that last longer are more likely to contribute to a customer's willingness to subscribe.

## Data Preparation

In order to prepare the dataset for modeling, some preprocessing steps were performed:
 - Encoding categorical variables: OneHotEncoder was used for categorical features like job, marital status, and education.
 - Feature selection: Variables like duration were dropped from training models as it was known only after contact with the customer.
 - Train-Test Split: 70% training and 30% testing data split.
 - The preprocessed data now contains encoded numerical values and is ready to be fed to machine learning models.

## Baseline Model Comparison

A dummy classifier was used as a baseline prior to training advanced models.

### Baseline Model Performance (Bar Graph Analysis)

![image](https://github.com/user-attachments/assets/7c5ed2d8-90d3-4c12-94dd-87d3fe07ddce)

- Performance was weak due to dataset imbalance with over-prediction of the majority class ("No").
- F1-score was below 0.50, which again supported a high misclassification rate.

The performance indicates the need for improved models and feature engineering.

### Heatmap of Feature Correlations
Here, we generated a correlation heatmap to verify how features correlate with each other. 

![image](https://github.com/user-attachments/assets/2d7e5232-dab2-42ce-aaad-e3bb5cb47403)

- Recall, Precision, and F1-score vary significantly across models, indicating different trade-offs.
- Accuracy remains high for most models, but precision and recall fluctuate, showing that some models favor either false positives or false negatives.
- Logistic Regression and Decision Trees performed better in terms of balancing precision and recall.

## Model Comparison

I trained and compared four classification models:

| Model                 | Train Accuracy | Test Accuracy | Time (s)  |
|-----------------------|---------------|--------------|----------|
| Logistic Regression  | 0.887527       | 0.886502     | 0.137    |
| KNN                 | 0.868528       | 0.865016     | 0.099    |
| Decision Tree       | 0.891472       | 0.881646     | 0.366    |
| SVM                 | 0.887618       | 0.886502     | 16.821   |

### Observations:
- Logistic Regression and SVM had the highest test accuracy (0.886502), which is a good indicator of good generalization.
- Decision Tree had the highest training accuracy (0.891472) but lower test accuracy (0.881646), which could be an indicator of overfitting.
- KNN performed the poorest on both training (0.868528) and test (0.865016) sets, which suggests that it may not be the best option for this dataset.
- SVM took much longer to train (16.821s), which makes it computationally costly, but it performed just as well as Logistic Regression.
- KNN had shortest training time of (0.099s) and was the fastest model but with lowest accuracy.

## Improving the Model

After developing baseline models, I focused on performance optimization via feature selection, hyperparameter tuning, and other transformations. The goal was to enhance the models without compromising interpretability and computational cost.

![image](https://github.com/user-attachments/assets/363c31c4-d163-4bb3-9d18-ba08b14c939c)

### Observations from Feature Importance:
- Contact method (cellular) was the strongest predictor of customer subscription probability.
- Job type (student, retired) was very strong, which suggests that career stage is an influence on financial decision-making.
- Credit default status (default_no) was also a strong predictor, with those with no default history more likely to subscribe.
- Marital status (single, divorced) and education level (basic.4y, university degree) were also strong contributors.

These insights informed our feature selection strategy such that the most significant features were retained for modeling. 

### Encoding and Data Transformation
Before fine-tuning models, categorical and numerical features were duly encoded by using OneHotEncoder for categorical variables and StandardScaler for numerical features so that all models could effectively ingest the data.

#### Steps:
- OneHotEncoding was applied to categorical features (job, marital, education, etc.).
- Scaling was applied to numerical features to normalize distributions.
- Feature transformation was checked, and it was ensured that categorical variables were encoded properly.

#### Final dataset shape after transformation:
- Training set: (X_train_encoded.shape)
- Test set: (X_test_encoded.shape)

### Hyperparameter Tuning
Next, we optimized the Decision Tree classifier by adjusting its depth and minimum sample split parameters. The objective was to strike a balance between model complexity and accuracy. 
| Max Depth                              | Test Accuracy |
|----------------------------------------|--------------|
| max_depth = 5, min_samples = 5         | 0.8863       |
|  max_depth = 10, min_samples = 10         | 0.8840       |
|  max_depth = None, min_samples = 2         | 0.8818       |

- Best Model: max_depth=5, min_samples_split=5 with 88.63% test accuracy.
- A deeper tree led to overfitting, corroborating the assertion that depth limitation improves generalization.

### Model Comparison After Improvements
After hyperparameter tuning, we reevaluated model performance on all classifiers.
| Model                | Train Accuracy | Test Accuracy | Time (s)  |
|----------------------|---------------|--------------|----------|
| Logistic Regression | 0.887527       | 0.886502     | 0.137    |
| KNN                 | 0.868528       | 0.865016     | 0.099    |
| Decision Tree       | 0.891472       | 0.881646     | 0.366    |
| SVM                 | 0.887618       | 0.886502     | 16.821   |

#### Observations from Improved Models:
- Logistic Regression came close, with fast training time and similar accuracy.
- Decision Trees improved after tuning, but overfitting risk remained an issue.
- KNN was the least accurate, as expected, since it performs poorly with big data.
- SVM worked correctly but was computationally expensive, taking a lot longer to train.

## Next Steps & Recommendations 
Based on the findings, the data is a massive class imbalance with the overwhelming majority of customers declining the offer of long-term deposit. This may have impacted model performance, especially recall and precision scores. To enhance predictive performance, measures in the next phase should include data balancing methods or undersampling the majority class in order to achieve a balanced dataset.

Additionally, decision tree model feature importance analysis revealed that contact method, level of education, and status of loan are critical drivers of customer response. Given that customers who were contacted through cellular phones showed a considerably greater rate of acceptance, the bank can enhance its marketing approach by prioritizing mobile over landline communication. Segmenting according to job type and education level would also allow more responsive segments of customers to be targeted.

To improve model performance, hyperparameter tuning should be extended beyond the attempted configurations to continue optimizing KNN and SVM models. Additional feature selection may also be employed to reduce noise and make models more efficient. A less computationally costly method, such as polynomial feature expansion in Logistic Regression or Decision Tree depth tuning, can also be attempted to improve classification without paying a high computational price. Lastly, business decisions must be informed by not just the results of the model but also external factors such as economic indicators because employment rate changes and consumer confidence indices can affect customer decisions.
