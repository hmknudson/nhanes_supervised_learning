# Predicting Overnight Hospitalization with NHANES Data

## 1. Introduction
The dataset in this capstone is an amalgamation of two individual datasets from the National Health and Nutrition Examination Survey (NHANES) from the 2015-2016 year. NHANES is a program of the National Center for Health Statistics, which is a subdivision of the Center for Disease Control and Prevention (CDC). NHANES data is collected every year and examines a nationally representative sample of approximately 5,000 people each time, who are located in 15 counties across the United States. The survey includes demographic, socioeconomic, and health-related questions, and the examination portion of the study includes medical, dental, and body measurements, as well as numerous lab tests. (For more information, see [NHANES](https://www.cdc.gov/nchs/nhanes/about_nhanes.htm).)

The dataset used here in this supervised learning capstone combines the NHANES 2015-16 "Demographic Variables and Sample Weights (DEMO_I)" dataset and the "Hospital Utilization and Access to Care (HUQ_I)" dataset. For more information about these datasets, the NHANES [DEMO_I](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm) and [HUQ_I](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HUQ_I.htm) documentation details all questions asked, responses to them, and general information on how the data was collected and processed.

To summarize, the DEMO_I data comes from the NHANES demographics questionnaire, which was asked in respondents' homes, using the Computer-Assisted Personal Interview (CAPI) system, which allowed to respondents to select English or Spanish as their language of choice, or alternatively, to request an interpreter. All adults answered questions directly, while a proxy answered questions on behalf of those under 16 or who were not able to answer for themselves. The HUQ_I (hospital utilization and access to care) data was collected in exactly the same way as DEMO_I.

## 2. Research Question
To what extent can incidences of overnight hospitalization be predicted with NHANES demographic and hospital utilization features from the 2015-2016 year?

## 3. Data

### About the data
The DEMO_I dataset comes from an XPT file found [here](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT), on the NHANES 2015-2016 page. Likewise, the HUQ_I dataset comes from an XPT file found [here](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HUQ_I.XPT). There are 9970 observations in both datasets, and their sequence numbers (row IDs) were identical, so they were easily concatenated. (Now add your screenshots here)

### Cleaning the data
The DEMO_I dataset originally had 47 columns. After cleaning, 29 remained. This is because columns containing more than 1000 nulls were removed, as this was deemed too large a percentage to reasonably impute data. After removing the columsn with more than 1000 null values, the columns with less than 1000 null values were dealt with on a case by case basis. Here, 3 columns were dropped because they were either copies of another, very similar column, or because the information in them was deemed irrelevant to the model (e.g., only 9 positive cases out of 9000 total observations). During this part of the cleaning process, 6 columns were imputed with the code '1000', so the missing data (which was recorded as nulls) could be used. 1000 was chosen, because it was sufficiently different from all other codes used in this research study, that it would not be mistaken for anything else.

The HUQ_I dataset originally had 10 columns. After cleaning, 8 columns remained, as 2 columns contained largely nulls and little else. Additionally, there were 3 columns contained a low enough amount of missing data (stored as null values) that it could be easily imputed with the code 1000, as mentioned previously.

When these two datasets were combined, the concatenated dataset was named demo_huq, and 2 more column that just had a small amount of missing data that were overlooked before were imputed with the code 1000 where needed. At this point, there were 37 columns, two of which were the 'seqn' columns that held the ID numbers for each row. There were also several more columns that were either copies of other columns or things that wouldn't be used in the model, such as a variable concerning the 'data release cycle' and another on 'interview/examination status' of the participants. After removing these, there were 26 columns left.

Then, all columns were re-coded to start with 0 rather than 1, as NHANES had made them, because it will be far easier for models to interpret data in binary format. After recoding, the age variable ('dmdhrage') was changed from categorical to continuous, both for ease of showing it with other variables in visualizations and also because separating the ages into distinct groups may make it easier to spot differences between age groups in the amount of overnight hospitalizations. The re-coded variables as well as the new categorical age variable (called 'dmdhrage_cat') were placed in a new dataframe called 'demo_huq_recode.' The dmdhrage_cat variable had a very small amount of null values present in it after being made categorical, so these were removed. Additionally, dmdhrage_cat was automatically made a 'category' type variable, but it needed to be a float like all the other variables, so the dataframe was copied into a new one called demo_huq_floatdf, and dmdhrage_cat was turned into a float64 type variable.

At this point, in the final dataset, there were 9954 observations and a total of 27 columns (because dmdhrage_cat was added). It was then time to prepare for building the models.

## 4. Methods

Here is a breakdown of the methods that will be used in model creation, optimization, and evaluation.

### Model creation
Separate the target from the categorical features (not going to use the continuous age variable). As a sidenote, it's clear the target's classes are imbalanced - the positive class (people who did stay in the hospital overnight) contains about 1/8 of the total observations, while the negative contains the other 7/8. However, I chose not to increase - and thereby even out - the amount of observations in the positive class (e.g., by using random over-sampling of the original datapoints), because I wanted to maintain the integrity of the original data, especially considering that it comes from a formal research study.

Run Spearman correlations between all of these features and the target. As the variables here are all categorical (including the target), Spearman will be used, since it is the non-parametric version of correlation tests (whereas Pearson's R is for normally distributed variables).

Set a threshold for correlations (e.g., +/- .1) and plan to use all features at or above said threshold.

Check these chosen features for multicollinearity (using both a correlation matrix and heatmap). If any are highly multicollinearity, remove one from each multicollinear pair.

Based on the specifications of this data (i.e., it is not normally distributed, all categorical, requires a classification task, and is a modest size), three different types of models will be trained - a logistic regression, a random forest, and a gradient boosting model (GBM). Each model will be evaluated in terms of its accuracy score and area under the ROC curve (auc score). The best performing model of these three will be chosen as the final model.

### Model optimization
1. The logistic regression will be tried in three forms, to try to find the best performing solver and type of regularization (if any). I'll use lbfgs (which is known to work well on fairly small datasets) with L2 regularization, saga (so I can try out L1 regularization), and lbfgs with no penalty. When I find which solver and type of regularization seems to work best for this data, I'll then iterate through several different values for max_iter to see if there is an optimal max_iter. For all of these logistic regression models, I'll use 8-fold cross-validation, and I'll use accuracy and auc scores as my scoring metrics in determining performance.

2. After the best logistic regression model from those I will have trained, I will run a feature importance analysis by retrieving the coefficients for each variable in the model, determining which are highest, and interpreting how each variable affects the target.

3. The random forest will first be run as a simple model with just the number of n_estimators specified, in order to get baseline scores for predictive power. Then, I will work on optimizing the random forest by manually running through numerous iterations with different n_estimators to find the optimal amount(s) of n_estimators for the model. Again, I'll be using 8-fold cross-validation, accuracy score, and auc score as evaluation metrics for all random forests. Once the best n_estimators is found, I'll run through numerous iterations with different max_depths, while keeping the n_estimators hyperparameter constant at what was found to be the optimal number. Once I have the best combination of n_estimators and max_depth, I will run a grid search to determine the best settings for other important hyperparameters (i.e., n_jobs, max_features, min_samples_leaf), leaving n_estimators and max_depth constant at the values I've determined to be optimal. In the event that the grid search ends up performing less well than my simpler iterations of random forests, I will simply go with the best random forest model I've achieved pre-grid search.

4. Then, a feature importance analysis will be run on the chosen random forest model, and levels of feature importance will be plotted as a bar graph.

5. The GBM will first be run as a simple model, only specifying the number of n_estimators to get a baseline (just as I began with the RF). Then, to be more efficient, I will use a randomized search to find the optimal set of hyperparameters from the following: learning_rate, n_estimators, max_depth, and max_features. Once the randomized search comes up with the best combination of hyperparameters, I will run a GBM with these exact hyperparameters, using 8-fold cross-validation and obtaining the accuracy and auc scores as evaluation metrics.

6. I'll then run a feature importance analysis on the GBM, making sure it falls generally within the same range as the RF's feature importance. If it doesn't, something might have gone wrong.

### Model evaluation
1. Supposing the feature importance analysis turns out find as expected, I can then compare the GBM's performance to the RF and the logistic regression. At this point, I will be able to choose the best model out of these three types and use it as my final model.

2. For whichever model I've ultimately chosen, I'll perform a statistical logistic regression on it, to figure out if I can make this model even better by potentially removing some features that may not be statistically significant. First, I'll run the statistical logistic regression with just a constant to determine the baseline; then, I'll run it with all features above a certain importance threshold included and see which are statistically significant to the model. If any are not, I'll remove them one at a time, to see how that removal affects the model. I will use the metrics of the entire model (AIC, BIC, p-value, log-likelihood, etc.) to determine whether the non-statistically significant feature should remain in the model or not. That is to say, if a feature is not statistically significant but still impacts the model positively overall, I will keep it in the model.

3. Then, once I've found the best combination of features to keep in the final model (and potentially optimized it even more), the statistical logistic regression coefficients will allow me to interpret the model better and determine what the odds are of a given outcome in the target, based on features' values.

## 5. Conclusions

### Recapping the project
In sum, three different types of classifiers were trained and tested. They were all run with 8-fold cross validation and judged on mean cross validation score and auc score.

The best logistic regression classifier achieved a mean cv score of .916 and auc score of .756. The coefficients for this model were obtained in order to explain the relative importance of each feature that went into this model.

The best performing random forest achieved a mean cv score of .901 and auc score of .992. The feature importance was obtained and plotted, to see which features were most important, but also to compare with the earlier logistic regression classifier.

The best gradient boosting model achived a mean cv score of .905 and auc score of .993. Feature importance was again obtained, plotted, and then compared with the random forest's feature importance, to ensure it fell within a fairly similar range.

The GBM won and was chosen as the final model to predict the target variable - overnight hospitalization. It had the highest predictive power of the three model types used here.

In order to fully consider the GBM's explanatory power as well, a statistical logist regression was run (only including the 12 most important features) to figure out if the model could be improved by removing some features that may not be statistically significant. Two features ended up being not statistically significant, so they were removed one at a time, to determine how their removal and their inclusion affected the model. Ultimately, keeping both these non-statistically significant features out improved the AIC, BIC, Log-Likelihood, and LL-Null values, and it made all other features statistically significant as well. Then, the GBM was run again, this time with only the 10 most important features (as the two that were not statistically significant were removed). Here, the mean cv score was .898, and the auc score was .983.

Although this GBM sacrificed some accuracy, it can still be thought of as a strong model, since it includes only the most important relationships to the target variable.

### Limitations
As always, there are a few limitations to this project. First, keeping the variables coded the same way NHANES had coded them - which is many cases is backwards (e.g., 0 means ‘yes’, 1 means ‘no’) - makes interpreting some coefficients and odds ratios confusing and was likely not the best choice.

Additionally, condensing the variables into smaller code ‘buckets’ rather than using dummy variables for each specific research code most likely affected the accuracy of at least some of the classifiers. In hindsight, I would use dummy variables to ensure every piece of information was getting taken into account separately by the models.

Finally, the race/ethnicity variable and the categorical age group variable were multicollinear, which wasn't caught right away, and which most likely affected the accuracy of the logistic regression model, if not any others.

Nonetheless, the final model achieved good predictive power and helped explain overnight hospitalization well. In a real life scenario, the demographic and health-related variables used here in the final GBM could be used to help predict overnight hospitalizations among other groups of people. Similarly, the odds ratios gathered from this GBM could help entities like insurance companies, hospitals, medical groups, social scientists (etc.) understand demographic factors that contribute to the likelihood of overnight hospitalization.

### Conclusion & Implications
One primary thing this project illustrates is that underlying or recurrent health conditions are a major influence on hospitalizations. Of course, this is an obvious assumption to make, but it can be confirmed empirically by noting that the variable that was most important in this analysis is the amount of times people have received healthcare in the past year. Obviously, a patient would seek healthcare more often if they have a chronic condition that flares up now and then or a condition like diabetes that is theoretically more manageable, but perhaps there could be compliance issues with insulin management, proper diet, etc.

Couching this variable in terms of compliance brings up a completely different set of issues. Namely, class and/or social status are major predictors of overnight hospitalization as well. As we saw in the feature importance analysis, income bracket and education level were the second and third most important features in the models (both random forest and GBM). Of course, a large chunk of the reason why people with higher incomes and higher education levels aren't staying in the hospital as often is due to having a better quality of life at home - there would be comfort and entertainment there, so these people most likely wouldn't choose to stay in the hospital unless it were absolutely necessary. Moreover, people with the highest levels of education may have more demanding jobs, stressful deadlines coming up, and they may be anxious return to those responsibilities rather than stay in the hospital. However, the fact remains that those with lower incomes and less education end up staying overnight in the hospital with greater frequency, while having a high income or high level of education is a protective factor against hospitalization.

Because of this fact, hospitals, insurance companies, or related non-profit organizations that are interested in decreasing rates of overnight hospitalization could work on developing more programs that target those demographics most at risk of overnight hospitalization. Additionally, focusing on increasing compliance with prescribed treatment plans could be a major goal of such programs, as hospitalization for some patients may be rendered unnecessary if timely preventive care is received and compliance is ensured.

### Future Research Possibilities
More qualitative (and/or quantitative) research is needed to actually determine how much of an issue compliance is with regard to overnight hospitalization. Similarly, more research could potentially tell us with more precision how income bracket, education level, and other social determinants actually influence the risk of overnight hospitalization and what methods would most effectively curb the rate of hospitalizations among those groups who are most at risk.

With regard to machine learning projects, it may be helpful to try building a model using only health science variables as opposed to the types of characteristic variables that featured heavily in this current project. It would be interesting to see if more biological, health-realated variables could make a model with stronger predictive power than models with demographic variables such as this current one.

In the same vein, it would be interesting to see how well this model can predict overnight hospitalization in other NHANES years. Using the same features but passing in unknown data would help determine more fully how strong of a model this is.
