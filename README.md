# Score-Prediction

## Introduction
A public educational institute in Singapore wants to build a model that can predict the students’ O-level mathematics examination scores to identify weaker students prior to the examination timely. Additional support can then be rendered to the students to ensure they are more prepared for the upcoming test.

The dataset used in this project is downloaded from https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db. This dataset belongs to AI Singapore (AISG) which they published under AIAP Technical Assessment Past Year Series (https://github.com/aisingapore/AIAP-Technical-Assessment-Past-Years-Series)

## Overview of the folder structure
- src
  - main.py
  - utils.py
  - config.py
- README.md
- eda.ipynb
- requirements.txt

## Script description
This ML project consists of the following pipeline workflow:
1. Extraction step: to extract the data from the respective data source and load it into a dataframe
2. Preprocessing step: to preprocess the input data (including any oversampling and scaling)
3. Tuning step: to tune the model with the best hyperparameters
4. Evaluation step: to evaluate the model based on our evaluation metric of interest
5. Explainability step: to gain insights into what features are important in influencing the ML prediction

Upon completion of the pipeline, the ML project will print the best ML model with the best evaluation metric (F1 micro) and also save the feature importance graph for model explainability.

## Summary of EDA
1. The data has ~3% missing data for final_test. As final_test is our target data of interest and we would like to make this prediction. ~3% is a small amount of missing data. Thus, we drop this data for training our model and we still have ~15k data entries to train and evaluate our ML model.
2. Features such as index and student_id are dropped as the data entries are too granular. bag_color is dropped as it is not meaningful or linked to your test score. wake_time and sleep_time are not meaningful as well so sleep_hours was derived instead based on wake_time and sleep_time as the no of sleeping hours will dictate if the students have sufficient rest. People with sufficient rest should have better focus.
3. Government dictates that students must have at least 15 year old before he/she can sit in for O'level test. Drop the students with age less than 15 year old as it means wrong input data.
4. The final test distribution is normal distributed.
5. attendance_rate and sleep_hours are highly correlated (r > 0.8). Thus, attendance_rate feature is dropped to avoid multicolinearity problems in the model. attendance_rate is chosen as feature to drop as it has missing data.
6. Final test score correlates positively with attendance_rate, sleep_hours, CCA_NONE, learning_style_visual and tuition_yes and negative low correlation with n_siblings. It seems like we need to let the students attend classess regularly, have sufficient rest, no CCA, learns by visual and have tuition to get them have better score.
7. One interesting finding is that final_test does not correlate positively with hours_per_week so it seems like students' score do not necessarily improve when they put in more hours.

## Summary Table of Data Preprocessing Step
| Feature name  | Data Preprocessing Performed |
| ------------- | ------------- |
| age  | Filter students to only those who are 15 year old and above  |
| index, student_id, bag_color, wake_time, sleep_time, attendance_rate  | Drop these features (explanation included on Summary of EDA) |
| sleep_hours  | Derive sleep_hours based on wake_time and sleep_time information  |
| Numerical features  | Standard Scaling all numerical features  |
| Categorical features  | Perform get_dummy variables for all categorical features  |

## Machine Learning model evaluation
The evaluation metric of interest is RMSE (Root Mean Square Error) as it is a regression problem. RMSE is chosen because we want to ensure we do not want our predicted score to be different much from the actual score so by using RMSE, it will penalise large errors more.
The best ML model achieved is XGBoost with RMSE score of 7.16.

ML evaluation metric table
| Model  | RMSE  |
| ------------- | ------------- |
| Linear Regression - ElasticNet  | 9.11  |
| Linear Regression - Ridge  | 9.11  |
| Linear Regression - Lasso  | 9.11  |
| XGBoost  | 7.16  |
| Neural Network  | 9.70  |

Based on the model explanability, the most importance features are as follow: <br />
By looking at the feature importance, the top important features that affect student score is mode_of_transport (walk and public transport), gender, tuition and learning_style by visual.
The company should focus on these features to identify weaker students.
![XGBoost Feature Importance](https://github.com/filbert11/Score-Prediction/blob/main/src/XGBoost%20Feature%20Importance.png)
