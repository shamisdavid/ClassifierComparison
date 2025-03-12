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

## Scatterplot Analysis
Scatterplots were used to examine correlations between attributes such as age vs. balance to determine if monetary factors play an important role in customer decision-making.

![image](https://github.com/user-attachments/assets/f26b6fc7-4ddf-431a-b070-4904ea428e94)

The plot revealed no correlation, suggesting that a number of different factors contribute to a customer's willingness to subscribe.

## Heatmap of Feature Correlations
Here, we generated a correlation heatmap to verify how features correlate with each other. 

![image](https://github.com/user-attachments/assets/2d7e5232-dab2-42ce-aaad-e3bb5cb47403)

There is high correlations between features such as employment change rate and consumer price index, indicating that the state of the economy affects customer decisions.

