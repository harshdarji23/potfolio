---
title: "Will Your Education Pay You Well?"
date: 2020-01-23
categories:
- Data Science
tags:
- data science
- analytics
thumbnailImagePosition: left
thumbnailImage: https://miro.medium.com/max/1750/0*MUWhjCFpm-r1cTPa.jpg
---

Wage analysis is a process of comparing the salaries based on the attributes attached to the employee. Of course, there are several factors like the company, location which contributes to the wage. However, we will analyze the Mid-Atlantic wage dataset, which is 
available [here](https://rdrr.io/cran/ISLR/man/Wage.html)
<!--more-->

For execution reason, I have utilized PySpark and Apache Spark Docker Jupyter Notebook, and you can utilize python and scikit or some other bundles.

We should peruse our information and perceive what it looks like:
```js
drop_cols = ['_c0', 'logwage', 'sex', 'region']
wage_df = spark.read.csv('/datasets/ISLR/Wage.csv', header=True, inferSchema=True).drop(*drop_cols)
training_df, validation_df, testing_df = wage_df.randomSplit([0.6, 0.3, 0.1], seed=0)
wage_df.limit(10).toPandas()
```
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/875/1*oG1ylILYg_JWyUPDDGHYtA.png" thumbnail="https://miro.medium.com/max/875/1*oG1ylILYg_JWyUPDDGHYtA.png" >}}
## Feature Engineering
We have a lot of categorical variables like Marital, race, education, job class, health, and health_ins, so we need to convert them into integer because ML models cannot work with categorical features. So, we use StringIndexer to convert a column to an index column. We also use One-hot encoding, which maps a categorical feature, represented as a label index, to a binary vector with at most a single one-value indicating the presence of a specific feature value from among the set of all feature values. Finally, we use vector assembler and merge all our variables and store them in the features column. We create a pipeline to keep our preprocessing code organized. Specifically, a pipeline bundles preprocessing and modeling steps so we can use the whole bundle as if it were a single step.

```js
from pyspark.ml import Pipeline

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

cols = ['maritl','race','education','jobclass','health','health_ins']

indexers = [

StringIndexer(inputCol=c, outputCol="{0}_index".format(c),stringOrderType="alphabetAsc",handleInvalid="error") for c in cols]
encoders = [

OneHotEncoder(

    inputCol=c + '_index',

    outputCol="{0}_feat".format(c),dropLast=False) 

for c in cols
]

numericCols = ['year','age']

assembler = VectorAssembler(

inputCols=numericCols+["{0}_feat".format(c) for c in cols],

outputCol="features"
)

pipe_feat = Pipeline(stages=indexers + encoders + [assembler]).fit(wage_df)

```
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/875/1*WtzE4n6eba2uQ6SSJKIpTA.png" thumbnail="https://miro.medium.com/max/875/1*WtzE4n6eba2uQ6SSJKIpTA.png" >}}
## Modeling — Random Forest

Now, we create three pipelines that contain three different random forest regressions that take in all features from the wage_df to predict wage
```js
from pyspark.ml.regression import RandomForestRegressor
rf_1 = RandomForestRegressor(featuresCol="features", labelCol="wage",maxDepth=1, numTrees=60, seed=1)
rf_2 = RandomForestRegressor(featuresCol="features", labelCol="wage",maxDepth=3, numTrees=40, seed=1)
rf_3 = RandomForestRegressor(featuresCol="features", labelCol="wage",maxDepth=6, numTrees=20, seed=1)

pipe_rf1=Pipeline(stages=[pipe_feat,rf_1]).fit(training_df)
pipe_rf2=Pipeline(stages=[pipe_feat,rf_2]).fit(training_df)
pipe_rf3=Pipeline(stages=[pipe_feat,rf_3]).fit(training_df)
```

## Evaluation
We will utilize RMSE to check the exhibition of our model. We register the RMSE on the validation data, and afterward, we relegate the best pipeline to the variable best_model. At long last, we utilize the best_model pipeline to figure the RMSE on the testing data.


```js
evaluator = evaluation.RegressionEvaluator(labelCol='wage', metricName='rmse')
rmse1=evaluator.evaluate(pipe_rf1.transform(validation_df))
print('RMSE For Model1:',rmse1)

rmse2=evaluator.evaluate(pipe_rf2.transform(validation_df))
print('RMSE For Model2:',rmse2)

rmse3=evaluator.evaluate(pipe_rf3.transform(validation_df))
print('RMSE For Model3:',rmse3)

best_model=pipe_rf3
```
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/528/1*oaHGER6PrQyEc5y9VYY5kQ.png" thumbnail="https://miro.medium.com/max/528/1*oaHGER6PrQyEc5y9VYY5kQ.png" >}}

We see that pipe_rf3 performs the best on validation data, so we will use it to check the performance on test data.

```js
RMSE_best=evaluator.evaluate(pipe_rf3.transform(testing_df))

final_model=Pipeline(stages=[pipe_feat,rf_3]).fit(wage_df)
```
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/875/1*JtL3vsDpWOsrnVgM31zYag.png" thumbnail="https://miro.medium.com/max/875/1*JtL3vsDpWOsrnVgM31zYag.png" >}}
## Inference
We make a pandas data frame feature_importance with the columns’ feature and importance. We will utilize the best model to decide the feature significance
Here, we want to give appropriate names such that someone can directly jump to inference and understand the entire model.
OK, this is an exceptionally moronic approach to get names like maritl_1._Never_Married! In the event that some can think of a well-characterized work, kindly ping me! Notwithstanding, we got what we needed

```js
maritl = pipe_feat.stages[0].getInputCol()

race = pipe_feat.stages[1].getInputCol()

education = pipe_feat.stages[2].getInputCol()

jobclass= pipe_feat.stages[3].getInputCol()

health = pipe_feat.stages[4].getInputCol()

health_ins = pipe_feat.stages[5].getInputCol()

#a=a.replace('index','1.Never_Married')

a = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[0].labels]
b = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[1].labels]
c = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[2].labels]
d = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[3].labels]
e = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[4].labels]
f = [i.replace(' ', '_') if isinstance(i, str) else i for i in pipe_feat.stages[5].labels]
a1=maritl+'_'+a[0]
a2=maritl+'_'+a[1]
a3=maritl+'_'+a[2]
a4=maritl+'_'+a[3]
a5=maritl+'_'+a[4]
b1=race+'_'+b[0]
b2=race+'_'+b[1]
b3=race+'_'+b[2]
b4=race+'_'+b[3]
c1=education+'_'+c[0]
c2=education+'_'+c[1]
c3=education+'_'+c[2]
c4=education+'_'+c[3]
c5=education+'_'+c[4]
d1=jobclass+'_'+d[0]
d2=jobclass+'_'+d[1]
e1=health+'_'+e[0]
e2=health+'_'+e[1]
f1=health_ins+'_'+f[0]
f2=health_ins+'_'+f[1]
feature_importance=pd.DataFrame(list(zip(['year','age',a1,a2,a3,a4,a5,b1,b2,b3,b4,c1,c2,c3,c4, c5,d1,d2,e1,e2,f1,f2 ], final_model.stages[-1].featureImportances. toArray())), columns = ['feature', 'importance']).sort_values('importance',ascending=False)
```
##### Feature Importance


{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/470/1*f3uPpaMz67vQqqie7ZKQDQ.png" thumbnail="https://miro.medium.com/max/470/1*f3uPpaMz67vQqqie7ZKQDQ.png" >}}
Random Forest has worked admirably of relegating significance to each component. Having an Advanced Degree and being old will bring about high pay, which bodes well. Additionally, marital status and race are given the least hugeness

<!--Lorem ipsum dolor sit amet, [test link]() consectetur adipiscing elit. **Strong text** pellentesque ligula commodo viverra vehicula. *Italic text* at ullamcorper enim. Morbi a euismod nibh. <u>Underline text</u> non elit nisl. ~~Deleted text~~ tristique, sem id condimentum tempus, metus lectus venenatis mauris, sit amet semper lorem felis a eros. Fusce egestas nibh at sagittis auctor. Sed ultricies ac arcu quis molestie. Donec dapibus nunc in nibh egestas, vitae volutpat sem iaculis. Curabitur sem tellus, elementum nec quam id, fermentum laoreet mi. Ut mollis ullamcorper turpis, vitae facilisis velit ultricies sit amet. Etiam laoreet dui odio, id tempus justo tincidunt id. Phasellus scelerisque nunc sed nunc ultricies accumsan.

Interdum et malesuada fames ac ante ipsum primis in faucibus. `Sed erat diam`, blandit eget felis aliquam, rhoncus varius urna. Donec tellus sapien, sodales eget ante vitae, feugiat ullamcorper urna. Praesent auctor dui vitae dapibus eleifend. Proin viverra mollis neque, ut ullamcorper elit posuere eget.


## List Types

### Definition List (dl)

<dl><dt>Definition List Title</dt><dd>This is a definition list division.</dd></dl>

### Ordered List (ol)

1. List Item 1
2. List Item 2
3. List Item 3

### Unordered List (ul)

- List Item 1
- List Item 2
- List Item 3

## Table

|  Header 1  | Header 2   | Header 3   |
|:----------:|------------|------------|
| Division 1 | Division 2 | Division 3 |
| Division 1 | Division 2 | Division 3 |
| Division 1 | Division 2 | Division 3 |
| Division 1 | Division 2 | Division 3 |

## Misc Stuff - abbr, acronym, sub, sup, etc.

Lorem <sup>superscript</sup> dolor <sub>subscript</sub> amet, consectetuer adipiscing <kdb>ctrl + c</kdb>. Nullam dignissim convallis est. Quisque aliquam. <cite>cite</cite>. Nunc iaculis suscipit dui.
 Nam
sit amet sem. Aliquam libero nisi, imperdiet at, tincidunt nec, gravida vehicula, nisl. Praesent mattis, massa quis luctus fermentum, turpis mi volutpat justo, eu volutpat enim diam eget metus. Maecenas ornare tortor. Donec sed tellus eget sapien fringilla nonummy. <acronym title="National Basketball Association">NBA</acronym> Mauris a ante. Suspendisse quam sem, consequat at, commodo vitae, feugiat in, nunc. Morbi imperdiet augue quis tellus.  <abbr title="Avenue">AVE</abbr>-->
