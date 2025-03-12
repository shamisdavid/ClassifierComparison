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

The plot readily indicates that the data is highly imbalanced and that the majority of customers declined the offer. The imbalance needs to be addressed in modeling.
Scatterplot Analysis
Scatterplots were used to examine correlations between attributes such as age vs. balance to determine if monetary factors play an important role in customer decision-making.

The plot revealed no correlation, suggesting that a number of different factors contribute to a customer's willingness to subscribe.

Heatmap of Feature Correlations
Here, we generated a correlation heatmap to verify how features correlate with each other. 

We observed high correlations between features such as employment change rate and consumer price index, indicating that the state of the economy affects customer decisions.

