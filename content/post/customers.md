---
title: "How to Identify Potential Customers Among the Crowd?"
slug: How-to-Identify-Potential-Customers-Among-the-Crowd
date: 2020-04-29
categories:
- Data Science
- Machine Learning
tags:
- data science
- customer segmentation
- kaggle
thumbnailImagePosition: left
thumbnailImage: https://miro.medium.com/max/875/0*6ZEe2u3ldlfq-pOz
---

A real-life data science task for a Mail-order sales company - The objective is to identify which individuals are most likely to respond to the campaign and become customers of the mail-order company.

<!--more-->
## Introduction
In this project, a [mail-order](https://en.wikipedia.org/wiki/Mail_order) sales company in Germany is interested in identifying segments of the general population to target with their marketing to grow. 
Demographics information has been provided (by Arvato Finacial Solutions through Udacity) for both the general population at large as well as for prior customers of the mail-order company to build a model of the customer base of the company. The target dataset contains demographics information for targets of a mailout marketing campaign.

## Data Description
The .csv files for this project are not provided because of a non-disclosure agreement with Arvato. The following are the files that were used:


- **Udacity_AZDIAS_052018.csv:** Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns)
- **Udacity_CUSTOMERS_052018.csv**: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- **Udacity_MAILOUT_052018_TRAIN.csv:** Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- **Udacity_MAILOUT_052018_TEST.csv:** Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
- **DIAS Attributes — Values 2017.xlsx:** Detailed information about the columns depicted in the files in alphabetical order. To learn more about the features, please click [here](https://github.com/harshdarji23/Arvato-Identifying-the-potential-customers/tree/master/Data).

## Data Preprocessing

- **Memory Reduction:** The .csv (Udacity_AZDIAS_052018) file holding demographic data of the general population was of the size of 2.5 GB, so I wrote a simple function to change the data types (int64 to int16) and reduce the memory usage by 78%. To learn more about the function and memory reduction, please click [here](https://www.kaggle.com/gemartin/load-data-reduce-memory-usage).
---
- **Data Understanding:** All the data files have the same features, so I decided to go with Udacity_AZDIAS_052018.csv file to understand the demographic data. I used DIAS Attributes — Values 2017.xlsx files to understand what each value means for the column. See the image below for an example:

{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*Vq7VcNwB9YbCaseTeFCbnA.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*Vq7VcNwB9YbCaseTeFCbnA.png" >}}



Here, the attribute **AGER_TYP** describes the best-ager typology. If you see the Value column above, it has values -1, 0, which means the meaning is unknown, and no classification is possible. On further, examination it's discovered that attributes having meaning **'unknown'/ 'no classification'** were missing values. So, I decided to fill such values with nan. Of course, there were many attributes that had values that were not known. To know the detailed analysis, please click [here](https://github.com/harshdarji23/Arvato-Identifying-the-potential-customers/tree/master/Jupyter%20Notebook). 
<!--haha-->
---
- **Handling Missing Values:** There were 366 features, so I decided to inspect the columns first after understanding the data(from pt.2) So, I calculated the % of missing values in each column and decided on a threshold 0f 30%, i.e., if columns contained more than 70% of missing values, I would simply drop those columns.

{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*kVIszoN8VE3Cydg99SegOg.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*kVIszoN8VE3Cydg99SegOg.png" >}}
On further inspection, I discovered that attributes starting with D19 were dropped. D19 when we look up in the DIAS Attributes - Values 2017.xlsx file shows that it contained transactional data (e.g., transactional activity based on the product group GUIDEBOOKS)

The next thing I did was to look missing values by rows, and I decided to remove the rows which contained more than 15 missing values. I here used Q3–1.5IQR to decide on the threshold of 15.
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*Zp7RjHa-CNhSJzTEXmu-Gw.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*Zp7RjHa-CNhSJzTEXmu-Gw.png" >}}
And the rest of the missing values were imputed by mean for simplicity reason.

- **Feature Engineering:** There were a lot of categorical variables, so I created dummy variables out of them if they had less than 10 levels. Now that all the features were numeric, we can apply any machine learning algorithm. But before that, I applied StandardScaler() to transform all the columns.
## Customer Segmentation
Here, the main goal was to use unsupervised learning methods to analyze attributes of established customers and the general population to create customer segments. The analysis describes parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

So, I used the PCA technique to capture the maximum variance in the data and reduce the dimensionality of data. I decided on a threshold of 50%.
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*3xMwaoNaisvv1VkhJr6Fhg.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*3xMwaoNaisvv1VkhJr6Fhg.png" >}}
36 features out of 366 were able to capture 50% of the variance. Following is the PCA Dimension 0 :
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*LHmTDdr514mWOHCH5hzQKQ.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*LHmTDdr514mWOHCH5hzQKQ.png" >}}

### Interpreting the 1st PCA component (top 10 features):
- **PLZ8_BAUMAX:** most common building-type within the PLZ8 (pos)-mainly >10 family homes
- **PLZ8_ANTG4:** number of >10 family houses in the PLZ8 (pos)-high share
- **PLZ8_ANTG3:** number of 6–10 family houses in the PLZ8 (pos)-high share
- **PLZ8_ANTG1:** number of 1–2 family houses in the PLZ8 (neg)-low share
- **MOBI_REGIO:** moving patterns (neg)-high mobility

So this group is **high mobility, large family area, crowed area, low-income.**

Next, I used data that was scaled using PCA to apply k-means clustering and identify customer segments groups. I used the elbow method to decide on the number of clusters, and I decided to go with 10.
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*jaZwb_EsUecWFqAXVriNnA.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*jaZwb_EsUecWFqAXVriNnA.png" >}}
I did the same transformation on the demographics data of the customer of the mail-order company Udacity_CUSTOMERS_052018.csv.

{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*ag5wCyguHlgEb5kM5CtECw.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*ag5wCyguHlgEb5kM5CtECw.png" >}}
Here, we can see that people belonging to cluster 1 and 8 are the ones who respond to a marketing campaign from a mail-order company and become customers. So, the marketing team should focus on such groups. The good news is that there are many people in Germany(Blue bar of 1 and 8) who fall into these groups.

### Marketing Predictions-Supervised Learning Model
Here, we will use the dataset containing the demographics data for individuals who were targets of a marketing campaign. The training dataset has the response of the customers, and we will use the ML model to learn the parameters and predict the response of customers in the test data.
- **Udacity_MAILOUT_052018_TRAIN.csv:** Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- **Udacity_MAILOUT_052018_TEST.csv:** Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

### Data Preparation
I have used the same cleaning function, which I used for the segmentation report. I have filled the missing values in numerical columns with mean, and I have created dummy variables for categorical columns. I have also performed a scaler transformation because I wanted to check the performance of different ML algorithms. However, the class in the dataset is an imbalance, the response from 98% in the training data is negative, and 2% is positive, so using recall(identifying potential customer) as the metric won't be right. Thus I have used ROC AUC as the metric to evaluate the performance. I have also added extra features from the customer dataset.
```js
test['CUSTOMER_GROUP'] = test['CUSTOMER_GROUP'].apply(lambda x:1 if x=='SINGLE_BUYER' else 0)
test['PRODUCT_GROUP1'] = test['PRODUCT_GROUP'].apply(lambda x:1 if 'FOOD' in x else 0)
test['PRODUCT_GROUP2'] = test['PRODUCT_GROUP'].apply(lambda x:1 if 'COSMETIC' in x else 0)
```
The best performing models were:

{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*kP43Ne2Xai-9BT56AkG-6Q.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*kP43Ne2Xai-9BT56AkG-6Q.png" >}}
---
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*LbnVUi--Mno4KqCp8Nz4hg.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*LbnVUi--Mno4KqCp8Nz4hg.png" >}}
I tried to optimize the hyperparameters of both the models, and Adaboostclasifer came at the top. The following are the parameters I used:

{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*5aRejUTn9LRC85DYXFXPvg.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*5aRejUTn9LRC85DYXFXPvg.png" >}}
### Prediction- Kaggle Competition
I submitted my prediction on Test data to the competition hosted on Kaggle, and I got a score of 0.79459, which is just 0.016 behind the 1st place on the leaderboard.
### Feature Importance
D19_SOZIALES: This is the feature related to a transaction that is not given in the data dictionary was the most important feature for a potential customer to respond to the marketing campaign.
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*OCyrtUHSkXkfbpiLQ--9dQ.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*OCyrtUHSkXkfbpiLQ--9dQ.png" >}}
### Conclusion
We used the demographic data of Germany and historical customer data of the marketing campaign to identify the best demographic group to target and thereby reducing marketing spend. We were successfully able to apply PCA and K-means clustering to identify customer segments. We used the Adaboost classifier with hyperparameter tuning to predict and classify if the customer will respond to a marketing campaign. A detailed analysis can be found on my [github](https://github.com/harshdarji23/Arvato-Identifying-the-potential-customers)

















<!--
If you want to add your site to this showcase, click [here](https://github.com/kakawait/hugo-tranquilpeak-theme/issues/new?title=Add%20my%20blog%20into%20the%20showcase&body=Hey,%20add%20my%20blog%20into%20the%20showcase:) and fill the following information:

- public url
- name (optional)
- description (optional)


## Eric Bouchut's blog

[![Eric Bouchut's blog](http://i.imgur.com/zQmKIKNm.png)](http://ericbouchut.com/)

## Robin Hu's blog

[![Robin Hu's blog](https://i.imgur.com/7SujaMam.png)](http://robinforest.net/)

## Zentechnista's blog

[![Zentechnista's blog](https://i.imgur.com/7zN7WMMm.png)](https://zentechnista.github.io/)

## Viajes Dendarii's blog

[![Viajes Dendarii's blog](https://i.imgur.com/tdXK3kYm.png)](https://dendarii.es)

## Wajahat Karim's blog

[![Wajahat Karim's blog](https://i.imgur.com/9BPoJvdm.png)](https://wajahatkarim.com/)

## Xiaoyun Yang's blog

[![Xiaoyun Yang's blog](https://i.imgur.com/vVRSvhpm.png)](http://xiaoyunyang.github.io/)

## Alfred E. Lin's blog

[![Alfred E. Lin's blog](https://i.imgur.com/lHwsvIJm.png)](http://alfredlin.com/)

## Philipp Gärtner's blog

[![Philipp Gärtner's blog](https://i.imgur.com/Sx6oXnSm.png)](https://philippgaertner.github.io/)

## Sagar Khatri's blog

[![Sagar Khatri's blog](https://i.imgur.com/edZ3PO9m.png)](https://www.ragasirtahk.tk/)

## Dr. Cruz Rincón's blog

[![Dr. Cruz Rincón's blog](https://i.imgur.com/XazQAolm.png)](https://www.cruzrincon.com.ve/)

## Björn Oettinghaus's blog

[![Björn Oettinghaus's blog](https://i.imgur.com/8vSMWIam.png)](https://www.datisticsblog.com/)

## Ivan Fadila Putra's blog

[![Ivan Fadila Putra's blog](https://i.imgur.com/r7tJa2Lm.png)](https://ffadilaputra.github.io/)

## BALLOON a.k.a. Fu-sen's blog

[![BALLOON a.k.a. Fu-sen's blog](https://i.imgur.com/ThaDHyfm.png)](https://balloon.gq/)

## Yue Hao's blog

[![Yue Hao's blog](https://i.imgur.com/CDDrTr4m.png)](https://yueyvettehao.netlify.com/)

## Adrian Riyadi's blog

[![Adrian Riyadi's blog](https://i.imgur.com/s6yB9lFm.png)](https://blog.adrian.id/)

## Vijay Mateti's blog

[![Vijay Mateti's blog](https://i.imgur.com/8LMItYSm.png)](https://vijaymateti.com/)

## Walid Benchaa's blog

[![Walid Benchaa's blog](https://i.imgur.com/8yn9DaOm.png)](https://rekkodo.gitlab.io/)

## Stella Wang's blog

[![Stella Wang's blog](https://i.imgur.com/F0jVpsOm.png)](https://hiwanglong.github.io/)

## Aditya Mangal's blog

[![Aditya Mangal's blog](https://i.imgur.com/FKrnNGlm.png)](https://www.adityamangal.com/) -->
