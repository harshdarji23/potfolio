---
title: "Building a Disaster response Web-Application"
date: 2020-05-03
categories:
- Data Science
tags:
- full-stack
- machine learning
- data science
- deployment
keywords:
- full-stack
- data science
- deployment
- data science
autoThumbnailImage: false
thumbnailImagePosition: "left"
thumbnailImage: "https://miro.medium.com/max/875/0*s6tzx9iUTs9Qr52q"
#coverImage: //d1u9biwaxjngwg.cloudfront.net/welcome-to-tranquilpeak/city.jpg
metaAlignment: center
---
In this project, we will be building a disaster response web application that will classify the message into different categories like medical supplies, food, or block road and direct them to the right organization to provide speedy recovery!

<!--more-->
<!--![Tranquilpeak](/img/showcase.png)
#Tranquilpeak is a gorgeous responsive theme for Hugo blog framework. It has many features and integrated services to improve user experience.-->
In 2019, there were a total of [409](https://www.statista.com/statistics/510959/number-of-natural-disasters-events-globally/) natural disasters worldwide. The irony is that we are right now in the middle of a global pandemic due to Covid19.  During a disaster or following the disaster, millions of people communicate either directly or via social media to get some help from the governmentor disaster relief and recovery services. If the affected person is tweeting it or even sending a message to the helpline service chances are that the message will be lost in the thousands of messages received. Sometimes it’s because a lot of people are just tweeting and very few people are needing help and organizations do not have enough time to filter out these many messages manually.
<!-- toc -->
---

<!--# Tranquilpeak-->

<!--[![Join the chat at https://gitter.im/LouisBarranqueiro/hexo-theme-tranquilpeak](https://badges.gitter.im/Join%20Chat.svg)](http s://gitter.im/LouisBarranqueiro/hexo-theme-tranquilpeak?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)-->

<!--A gorgeous responsive theme for Hugo blog framework

[![Tranquilpeak](/img/showcase.png)](https://tranquilpeak.kakawait.com)-->
<p></p>
<br>	


<!--[Unsplash](https://images.unsplash.com/photo-1475776408506-9a5371e7a068?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60)-->

{{< image classes="fancybox fig-100" src="https://images.unsplash.com/photo-1475776408506-9a5371e7a068?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60" thumbnail="https://images.unsplash.com/photo-1475776408506-9a5371e7a068?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60" >}}
## Data Overview

<!--**ATTENTION** during *alpha* or *beta* [versions](https://github.com/kakawait/hugo-tranquilpeak-theme/milestones) breaking changes are possible on config file.

You can track [breaking changes label](https://github.com/kakawait/hugo-tranquilpeak-theme/issues?q=is%3Aissue+is%3Aopen+label%3A%22breaking+changes%22).

How can I migrate my current version? Please read [CHANGELOG.md](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/CHANGELOG.md).-->
We will be analyzing real messages that were sent during disaster events. The data was collected by [Figure Eight](https://appen.com/) and provided by [Udacity](https://classroom.udacity.com/courses/ud257), a big thank you to them. Let’s look at the data description:

<!--## Credits

*Hugo* version of Tranquilpeak is a based on original *Hexo* version https://github.com/LouisBarranqueiro/hexo-theme-tranquilpeak. This version is simply a port to *Hugo* static site generator.

Please all the credit should be attributed to [original *Hexo* version](https://github.com/LouisBarranqueiro/hexo-theme-tranquilpeak) and its author [Louis Barranqueiro](https://github.com/LouisBarranqueiro).

*Hugo* version keeps every `.js` and `.css` files untouched from original *Hexo* version in order to enjoy futur original *Hexo* version updates or features!-->
- [messages.csv](https://github.com/harshdarji23/Disaster-Response-WebApplication/blob/master/workspace/data/disaster_messages.csv): Contains the id, message that was sent and genre i.e the method (direct, tweet..) the message was sent.
- [categories.csv](https://github.com/harshdarji23/Disaster-Response-WebApplication/blob/master/workspace/data/disaster_categories.csv): Contains the id and the categories (related, offer, medical assistance..) the message belonged to.

## ETL Pipeline
So, in this part, we will merge two datasets, **messages.csv**, and **categories.csv** on the common id column. The category column in the id categories.csv is in a string format so we need to create columns for each category. Then we will remove duplicates and load the dataset the transformed data into the database hosted using the SQLAlchemy library.
<!--
- [General](#general)
- [Features](#features)
- [Quick start](#quick-start)
- [Demo](#demo)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Showcase](#showcase)
- [License](#license)-->
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/840/1*eYtOsK4PowFQrl0FE7-5iQ.png" thumbnail="https://miro.medium.com/max/840/1*eYtOsK4PowFQrl0FE7-5iQ.png" >}}
---
The categories are of the form
---
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/520/1*_0t7hCJT_tCU6Pn85e9hLQ.png" thumbnail="https://miro.medium.com/max/520/1*_0t7hCJT_tCU6Pn85e9hLQ.png" >}}
---
After transformation
---
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/726/1*t4CrRIQArf11E7LL2a_UzA.png" thumbnail="https://miro.medium.com/max/726/1*t4CrRIQArf11E7LL2a_UzA.png" >}}
---
Load data into database
---
And we finally load the transformed data into the database: disaster.db. <br>
Checkout the code for the entire ETL pipeline [here](https://github.com/harshdarji23/Disaster-Response-WebApplication/blob/master/Jupyter%20Notebooks/ETL%20Pipeline%20Preparation.ipynb)

```js
engine = create_engine('sqlite:///disaster.db');
df.to_sql('disaster_response', engine, index=False);
```
---
ML Pipeline
---
Here, we will load the dataset from the disaster.db database. Our main task is to convert the messages into tokens so that they can be interpreted. So, we create a function that will remove punctuations, tokenize words, remove stop words, and perform lemmatization.

```js
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize and remove punctuation
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens
```

So, this is what our function will do
---
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/669/1*D2hkb6qSMwaKPGodlp8APQ.png" thumbnail="https://miro.medium.com/max/669/1*D2hkb6qSMwaKPGodlp8APQ.png" >}}

These words make sense but they cannot be understood by the ML model. So, we will use countVectorizer and tfidf transformer to transform the tokens into features(integers) and we use a simple Random Forest Classifier to fit the training data.
```js
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize))
    , ('tfidf', TfidfTransformer())
    , ('clf', MultiOutputClassifier(RandomForestClassifier()))])
pipeline.fit(X_train, Y_train)
```

For evaluating our model, we will be using the F-1 score as both False Negatives and False positives are important to us i.e. if we fail to predict the right category of the message then we won’t be able to provide right assistance and if we wrongly predict the category of the message we will be wasting our time.
The Random Forest classifier gives us an F-1 score of 0.44. The main reason behind the low score is that the categories are highly imbalanced. The distribution of the categories is as follows:

---
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/875/1*k01VKOQS26M90yNbw9xcow.png" thumbnail="https://miro.medium.com/max/875/1*k01VKOQS26M90yNbw9xcow.png" >}}
Let’s improve the model using some different ML model and hyperparameter tuning. So, after doing a GridSearchCV to find the best parameter of the Random Forest model we were able to increase the F-1 score to 0.51. Next, we train AdaBoost classifier and we were able to improve the F-1 score to 0.59

```js
#https://medium.com/swlh/the-hyperparameter-cheat-sheet-770f1fed32ff
pipeline_ada = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
    ))
])
parameters_ada = {
    'clf__estimator__learning_rate': [0.1, 0.3],
    'clf__estimator__n_estimators': [100, 200]
}
cv_ada = GridSearchCV(estimator=pipeline_ada, param_grid=parameters_ada, cv=3, scoring='f1_weighted', verbose=3)
```
{{< cta-button “Sign up for free!” “https://sensr.net/auth/users/sign_up 5” >}}
We save this model as a pickle file so that we do not need to train it again. Code available [here](https://github.com/harshdarji23/Disaster-Response-WebApplication/blob/master/Jupyter%20Notebooks/ML%20Pipeline%20Preparation.ipynb)

```js
pickle.dump(cv_ada, open('disaster_ada_model.sav', 'wb'))
```

Flask Application
We will create a train_classifer.py to create functions that will transform, load, build, and save the model. Basically, we will use the ETL pipeline and ML pipeline. We will create a folder named app that will contain a master.html file which will be the front end and run.py that will run behind to perform computation.

## Demo


{{< twitter 1256761506949222407 >}}



## Conclusion
Build a full-stack multi-output ML web application to classify messages sent during disasters into different categories and provide quick assistance from different disaster relief organizations. You can run this application on your own computer by following the instructions on my [github](https://github.com/harshdarji23/Disaster-Response-WebApplication).


---
Thank you for reading!
<!--## General

- **Authors**: [Louis Barranqueiro (LouisBarranqueiro)](https://github.com/LouisBarranqueiro) and [Thibaud Leprêtre (kakawait)](https://github.com/kakawait)
- **Version**: 0.4.8-BETA (based on Hexo version 1.10.0)
- **Compatibility**: Hugo v0.53

## Features

**General features:**

- Fully responsive
- Optimized for tablets & mobiles
- Configurable menu of the sidebar
- Pages to filter tags, categories and archives
- Background cover image
- Beautiful about page
- Support Open Graph protocol
- Easily customizable (fonts, colors, layout elements, code coloration, etc..)
- Documentations
- Support internationalization (i18)

**Posts features:**

- Thumbnail image
- Cover image
- Responsive videos & images
- Sharing options
- Navigation menu
- GitHub theme for code highlighting (customizable)
- Image gallery
- Tags for images (FancyBox), wide images, tabbed code blocks, highlighted text, alerts
- Table of contents

**Integrated services:**

- Disqus
- Google analytics
- Gravatar
- Facebook Insights

### Missing features from original *Hexo* version

- [ ] Baidu analytics
- [ ] Algolia (https://github.com/kakawait/hugo-tranquilpeak-theme/issues/8)
- [ ] Pagination custumization `tagPagination`, `categoryPagination` and `archivePagination` (https://github.com/kakawait/hugo-tranquilpeak-theme/issues/17)

**ATTENTION** following features will not be possible due to *Hugo* limitations

- Archives pages by years `/archives/2015`
- Archives pages by month `/archives/2015/01`

## Quick start

**Please read [user documentation](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/user.md), it's short and useful to discover all features and possibilities of the theme, especially the  [writing posts](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/user.md#writing-posts) section**

### For people who want to use the original version of Tranquilpeak without modifications (users)

Go to the directory where you have your Hugo site and run:

```shell
mkdir themes
cd themes
git clone https://github.com/kakawait/hugo-tranquilpeak-theme.git
```

After installing the Tranquilpeak theme successfully, we recommend you to take a look at the [exampleSite](exampleSite) directory. You will find a working Hugo site configured with the Tranquilpeak theme that you can use as a starting point for your site.

First, let's take a look at the [config.toml](exampleSite/config.tom). It will be useful to learn how to customize your site. Feel free to play around with the settings.

More information on [user documentation](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/user.md) to install and configure the theme

### For people who want to create their own version of tranquilpeak (developers)

1. Run `git clone https://github.com/kakawait/hugo-tranquilpeak-theme.git`
2. Follow [developer documentation](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/developer.md) to edit and build the theme

## Demo

Check out Tranquilpeak theme in live : [hugo-tranquilpeak-theme demo](https://tranquilpeak.kakawait.com)

## Showcase

Checkout showcase https://github.com/kakawait/hugo-tranquilpeak-theme/wiki/Showcase

### How can I add my site to the showcase

**Click [here](https://github.com/kakawait/hugo-tranquilpeak-theme/issues/new?title=Add%20my%20blog%20into%20the%20showcase&body=Hey,%20add%20my%20blog%20into%20the%20showcase:) to add your blog into the showcase.**

Please fill the following information:

1. public url
2. name (optional)
3. description (optional)

## Documentation

If it's your first time using Hugo, please check [Hugo official documentation](https://gohugo.io/overview/introduction/)

### For users

To install and configure the theme, consult the following documentation : [user documentation](https://githubh.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/user.md)

### For developers

To understand the code, the workflow and edit the theme, consult the following documentation : [developer documentation](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/docs/developer.md)

## Contributing

All kinds of contributions (enhancements, new features, documentation & code improvements, issues & bugs reporting) are welcome.

As explained on [Credits](#credits):

> *Hugo* version keeps every `.js` and `.css` files untouched from original *Hexo* version in order to enjoy futur original *Hexo* version updates or features!

That mean I would keep a stronge dependency with original *Hexo* theme. Thus if you want to suggest any modifications on `.css` or `.js` files **I will submit those changes to original *Hexo* theme** (except if it's really specific to *Hugo* bugs that is not present on *Hexo*).

## License

hugo-tranquilpeak-theme is released under the terms of the [GNU General Public License v3.0](https://github.com/kakawait/hugo-tranquilpeak-theme/blob/master/LICENSE).-->


