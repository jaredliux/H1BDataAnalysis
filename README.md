# H1BDataAnalysis - v1
This is a school project I have done with a teammate for the course 95885 Data Science and Big Data at CMU. This notebook includes discovering potential factors that affect H1-B visa applications. and provides a machine learning model to classify visa status automatically. Detailed descriptions are in H1B visa analysis.pptx and H1B Data Analysis_final.ipynb in folder v1. <br>

## Problem Statement
H-1B visa – most common visa for international students to apply if they want to hold full time jobs in the US after graduation. It is important to have a good sense of what is important during application process. Therefore, this analysis aims to uncover insights to help employers and international students better understand the process. More specifically, <br>
#### (1) This analysis builds a machine learning model to predict whether or not an H1B application will be approved or denied, and <br>
#### (2) From the machine learning model, we can see which factors are more important in this process (Salary? Occupation?).
<br>

## Key Insights
* An application’s truthfulness, completeness, and detailedness are the most influencing factors.
* Employers must compensate foreign workers fairly.
* H-1B visa demands highly-skilled applicants.
<br>

## Data Collection and Preprocessing
### a. Data Source
We collected the visa application status for the past three years to get a general idea of the recent situation. The data source for this project is U.S. Department of Labor. Datasets and detailed descriptions for all features can be found in https://www.dol.gov/agencies/eta/foreign-labor/performance.

### b. Data Format
The datasets are in .xlsx format. After reading into Pandas dataframe, there are near 2 million rows and around 260 features. Number of features varies from year to year. The label we are going to predict here is **CASE_STATUS**, which is categorical variable - *CERTIFIED*, *DENIED*, *WITHDRAWN*, and *CERTIFIED-WITHDRAWN*. We don't consider the last two cases in our analysis.

### c. Data Cleaning and Feature Engineering
Having removed missing data (since the dataset is fairly large), we handpicked 26 columns, which exist in all three years' data and include few missing data. Then, we created and reformated new features based on our understanding, and looked at data visualizations between variables. Following relationships are shown in the plots.
* Certified rate keeps increasing in recent years.
* Certified rate varies drastically among occupations, with computer, business and engineering related positions at the top. More popular occupations – more opportunities for international students.
* Certified rate does not vary much among industries.
* Companies that are H-1B dependent and with attorneys have advantages.
* If applicant’s wage is lower than prevailing wage, the chance to be denied is much higher.

## Model Training and Evaluation
### a. Challenges
* Too many irrelevant features. Solution: We only included 9 features in our model training process.
* Highly imbalanced dataset. Solution: We undersampled CERTIFIED data, resulting in 46000 rows of data. (Cavaet: This loses a lot of information, therefore, it might not be the best solution. However, upsampling is also not feasible, since there will result in too many duplicate data. We can utilize anomaly detection algorithms to address this in the future.)

### b. Input data
The size of input data is 46000 by 9, with 50% for both labels. The train-test split is 80% - 20%.

### c. Model Training
We use cross validation to do hyperparamter tuning and choose Logistic Regression, Decision Tree, Random Forest, XGBoost. The results for validation sets are in the following table.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| LogR  | 0.795    | 0.815     | 0.762  | 0.788    |
| DT    | 0.868    | 0.912     | 0.814  | 0.860    |
| RF    | 0.938    | 0.976     | 0.897  | 0.935    |
| XGB   | 0.891    | 0.942     | 0.833  | 0.884    |

Here, Logistic Regression serves as the baseline model. We can see Random Forest has the highest numbers among other models. So, we train Random Forest on all training data and evaluate on test set. The results are as below.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| RF    | 0.863    | 0.893     | 0.827  | 0.859    |

A little drop is expected. We can increase the number of observations to increase the performance.

### d. Feature Importance
One of the advantages for tree-based models is the explainability through feature importance. In our model, we can check faetures with top feature importance are *individual-industry wage difference*, *hourly wage*, *computer-related and technical positions and industries*, *H-1B dependent*, and *agents represented*.

## Future Work
There are still a lot of room to improve. Here, I propose some ideas.
* Huge amount of effort could be invested in feature engineering as we deepen our understanding on this subject.
* Gather more data about the employers and education information. As more advanced education could be helpful in the process.
* Use anomaly detection algorithms to make use of most data, which can possibly lead to better results.
