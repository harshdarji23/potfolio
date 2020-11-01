---
title: "Building the SMS spam detector"
date: 2019-12-24
categories:
  - Data Science
tags:
  - data science
  - machine learning
  - cross-validation
thumbnailImagePosition: left
thumbnailImage: https://miro.medium.com/max/875/0*GB1iB6fPoZjT3-zC.jpg
---
In this blog post, we are going to develop an SMS spam detector using logistic regression and pySpark. We will predict whether an SMS text is spam or not Brace yourselves the spam is coming!!!
<!--more-->

## Dataset
Text file can be downloaded from [here](https://github.com/harshdarji23/Sms-spam-detector/blob/master/sms_span_df.csv). This is what our dataset looks like:


{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/519/1*rBzjFDqCt5asPBD4tRlZQA.png" thumbnail="https://miro.medium.com/max/519/1*rBzjFDqCt5asPBD4tRlZQA.png" >}}

We are using pySpark for distributed computing and we will create a machine learning pipeline to automate the workflow. We see that the “type” column contains categorical data, so the first step is to convert the contents of the “type” column to numeric attributes. We encode because logistic regression cannot operate with categorical data.

```js
sms_spam_df.createOrReplaceTempView('temp')
sms_spam2_df=spark.sql('select case type when "spam" then 1.0 else 0 end as type, text from temp')
sms_spam2_df.show()
```
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/325/1*Fvdww9HRGzIvpXeXgZkvCw.png" thumbnail="https://miro.medium.com/max/325/1*Fvdww9HRGzIvpXeXgZkvCw.png" >}}


Now, we will create a pipeline that will combine a Tokenizer, CounterVectorizer, and an IDF estimator to compute the TF-IDF vectors of each SMS.

```js

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")
from pyspark.ml.feature import CountVectorizer
countvectorizer = CountVectorizer (inputCol="words", outputCol="tf")
from pyspark.ml.feature import IDF
idf = IDF(inputCol="tf", outputCol="tfidf")
tfidf_pipeline = Pipeline(stages=[tokenizer,countvectorizer,idf]).fit(sms_spam2_df)
```
- **Tokenizer:** It creates words(tokens) from sentences for each SMS
- **Countvectorizer:** It counts the number of times a token shows up in the document and uses this value as its weight.
- **TF-IDF Vectorizer:** TF-IDF stands for “term frequency-inverse document frequency”, meaning the weight assigned to each token not only depends on its frequency in 
a document but also how recurrent that term is in the entire corpora.


{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/875/1*kSQ-mULHLeYYk2aJubpvFg.png" thumbnail="https://miro.medium.com/max/875/1*kSQ-mULHLeYYk2aJubpvFg.png" >}}

Now that we have our sentences in TF-IDF form, let us create ML pipelines where the first stage is the tfidf_pipeline created above and the second stage is a LogisticRegression model with different regularization parameters (𝜆) and elastic net mixture (𝛼)


### Compare Models

```js
training_df, validation_df, testing_df = sms_spam2_df.randomSplit([0.6, 0.3, 0.1], seed=0)
lr1 = classification.LogisticRegression(regParam=0, elasticNetParam=0, labelCol = 'type', featuresCol = 'tfidf')
lr_pipeline1 = Pipeline(stages=[tfidf_pipeline, lr1]).fit(training_df)
lr2 = classification.LogisticRegression(regParam=0.02, elasticNetParam=0.2, labelCol = 'type', featuresCol = 'tfidf')
lr_pipeline2 = Pipeline(stages=[tfidf_pipeline, lr2]).fit(training_df)
lr3 = classification.LogisticRegression(regParam=0.1, elasticNetParam=0.4, labelCol = 'type', featuresCol = 'tfidf')
lr_pipeline3 = Pipeline(stages=[tfidf_pipeline, lr3]).fit(training_df)
```

We use cross-validation because it helps us better use our data, and it gives much more information about our algorithm’s performance.

```js
evaluator = evaluation.BinaryClassificationEvaluator(labelCol='type')
AUC1 = evaluator.evaluate(lr_pipeline1.transform(validation_df))
print("Model 1 AUC: ", AUC1)
AUC2 = evaluator.evaluate(lr_pipeline2.transform(validation_df))
print("Model 2 AUC: ", AUC2)
AUC3 = evaluator.evaluate(lr_pipeline3.transform(validation_df))
print("Model 3 AUC: ", AUC3)
```

{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/435/1*5NRuAXY7KLhz3NGjGjGAVg.png" thumbnail="https://miro.medium.com/max/435/1*5NRuAXY7KLhz3NGjGjGAVg.png" >}}

We see that model 2 with pipeline lr_pipeline2 containing regParam=0.02, elasticNetParam=0.2 performs best on the validation data so we fit this pipeline to our test data and find out it’s AUC on test data.

```js 
AUC_best = evaluator.evaluate(lr_pipeline2.transform(testing_df))
```

## Inference

Now we will use the pipeline 2 fitted above (lr_pipeline2) to create Pandas data frames that contain the most negative words and the most positive words.
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/365/1*-QvGr-NsXVgGz85klwej0w.png" thumbnail="https://miro.medium.com/max/365/1*-QvGr-NsXVgGz85klwej0w.png" >}}
{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/253/1*rLlhLzAdpW-hVS_CrIB7uA.png" thumbnail="https://miro.medium.com/max/253/1*rLlhLzAdpW-hVS_CrIB7uA.png" >}}


## Conclusion

We converted our text into tokens and TF-IDF vectors, played around with parameters of logistic regression, and evaluated our model using AUC metric on the validation data. Finally, based on the performance of the model on the validation data, we fitted the model on the test data, and measured the model’s performance. Thus, we developed a spam detector using regularized logistic regression.

##### Can we improve the performance of our model?
The performance of the model can be improved by feature engineering on the data. Typical spam messages contain words that are upper case. So we create a data frame sms_spam3_df where we add a new column has_uppercase which contains an integer 1 if the first sequence of uppercase letters is longer or equal to 3 and an integer 0 otherwise

```js
from pyspark.sql.functions import regexp_extract, col,length
sms_spam3_df=sms_spam2_df.select('type','text', regexp_extract(col('text'), '([A-Z]{3,})', 1).alias('has_upppercase'))
sms_spam3_df = sms_spam3_df.select('type','text', fn.when(fn.length(fn.col('has_upppercase')) >= 3, 1).otherwise(0).alias('has_uppercase'))
```


{{< image classes="fancybox fig-100" src="https://miro.medium.com/max/489/1*lxvgiVePSB82fk0M2kMnTQ.png" thumbnail="https://miro.medium.com/max/489/1*lxvgiVePSB82fk0M2kMnTQ.png" >}}

Let’s see what our data frame looks like:
{{< image classes="fancybox fig-100" src="https://cdn-images-1.medium.com/max/1000/1*Vq7VcNwB9YbCaseTeFCbnA.png" thumbnail="https://cdn-images-1.medium.com/max/1000/1*Vq7VcNwB9YbCaseTeFCbnA.png" >}}


```js
from pyspark.ml.feature import Tokenizer
tokenizer_df3 = Tokenizer(inputCol="text", outputCol="words")
from pyspark.ml.feature import CountVectorizer,VectorAssembler
ountvectorizer_df3 = CountVectorizer(inputCol="words", outputCol="tf")
from pyspark.ml.feature import IDF
idf_3 = IDF(inputCol="tf", outputCol="tfidf")
assembler = VectorAssembler(inputCols=["has_uppercase", "tfidf"],outputCol="features")
tfidf_pipeline_upper = Pipeline(stages=[tokenizer_df3,countvectorizer_df3,idf_3,assembler]).fit(sms_spam3_df)
df_upper = tfidf_pipeline_upper.transform(sms_spam3_df)
#!pip install git+https://github.com/daniel-acuna/pyspark_pipes.git
from pyspark_pipes import pipe
scaled_model=pipe(feature.VectorAssembler(inputCols=['has_uppercase','tfidf']),feature.MaxAbsScaler(),classification.LogisticRegression(regParam=0.2, elasticNetParam=0.1, labelCol = 'type'))
scaled_model_fitted = scaled_model.fit(df_upper)
```

Now, that we have two columns of text, and has_uppercase, we have to tokenize and create TF-IDF of the text and then merge with has_uppercaase column using vector assembler. We create a pipeline that will merge the two columns, perform feature scaling using MaxAbsScaler, and run a logistic regression model (lr2 regularization parameter 𝜆=0.2 and elastic net mixture 𝛼=0.1) that performs best on the above data frame.

##### Is has_uppercase, a feature that is positively or negative related to an SMS being spam?

```js
my_coeff=scaled_model_fitted.stages[-1].coefficients
has_uppercase_coeff = my_coeff.toArray()[0]
print('has_uppercase feature is positively related to an SMS being spam with a coefficient of:',has_uppercase_coeff)
```

We fetch the coefficient of has_uppercase feature from the pipeline and it comes out to be 0.9289. Thus, has_uppercase is positively related to an SMS being spam

#### What is the ratio of the coefficient of has_uppercase to the biggest positive tfidf coefficient?

```js
max_coeff=my_coeff.toArray().max()
my_ratio=has_uppercase_coeff/max_coeff
print('The ratio of the coefficient of has_uppercase to the biggest positive tfidf coefficient is :',my_ratio)
```

The max-coefficient of Tfidf comes out to be 2,01, so the ratio of the coefficient of has_uppercase to the biggest positive tfidf coefficient is 0.46

---

Thank you for reading! Feedbacks are highly appreciated.
<!--
# Tabbed code block

{{< tabbed-codeblock tabbed_codeblock >}}
<!-- tab js -->
<!--function $initHighlight(block, flags) {
  try {
    if (block.className.search(/\bno\-highlight\b/) != -1)
      return processBlock(block.function, true, 0x0F) + ' class=""';
  } catch (e) {
    /* handle exception */
    var e4x =
        <div>Example
            <p>1234</p></div>;
  }
  for (var i = 0 / 2; i < classes.length; i++) { // "0 / 2" should not be parsed as regexp
    if (checkCondition(classes[i]) === undefined)
      return /\d+[\s/]/g;
  }
  console.log(Array.every(classes, Boolean));
}
<!-- endtab -->
<!-- tab css -->
<!--@media screen and (-webkit-min-device-pixel-ratio: 0) {
  body:first-of-type pre::after {
    content: 'highlight: ' attr(class);
  }
  body {
    background: linear-gradient(45deg, blue, red);
  }
}

@import url('print.css');
@page:right {
 margin: 1cm 2cm 1.3cm 4cm;
}

@font-face {
  font-family: Chunkfive; src: url('Chunkfive.otf');
}

div.text,
#content,
li[lang=ru] {
  font: Tahoma, Chunkfive, sans-serif;
  background: url('hatch.png') /* wtf? */;  color: #F0F0F0 !important;
  width: 100%;
}
<!-- endtab -->
<!-- tab html -->
<!--<?xml version="1.0"?>
<response value="ok" xml:lang="en">
  <text>Ok</text>
  <comment html_allowed="true"/>
  <ns1:description><![CDATA[
  CDATA is <not> magical.
  ]]></ns1:description>
  <a></a> <a/>
</response>


<!DOCTYPE html>
<title>Title</title>

<style>body {width: 500px;}</style>

<script type="application/javascript">
  function $init() {return true;}
</script>

<body>
  <p checked class="title" id='title'>Title</p>
  <!-- here goes the rest of the page -->
<!--</body>
<!-- endtab -->
<!--{{< /tabbed-codeblock >}}

# ApacheConf

{{< codeblock "apache.conf" "apacheConf" "http://underscorejs.org/#compact" "apache.conf" >}}
# rewrite`s rules for wordpress pretty url
LoadModule rewrite_module  modules/mod_rewrite.so
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . index.php [NC,L]

ExpiresActive On
ExpiresByType application/x-javascript  "access plus 1 days"

Order Deny,Allow
Allow from All

<Location /maps/>
  RewriteMap map txt:map.txt
  RewriteMap lower int:tolower
  RewriteCond %{REQUEST_URI} ^/([^/.]+)\.html$ [NC]
  RewriteCond ${map:${lower:%1}|NOT_FOUND} !NOT_FOUND
  RewriteRule .? /index.php?q=${map:${lower:%1}} [NC,L]
</Location>
{{< /codeblock >}}

# Bash

{{< codeblock "test.bash" "bash" "http://underscorejs.org/#compact" "test.bash" >}}

#!/bin/bash

###### BEGIN CONFIG
ACCEPTED_HOSTS="/root/.hag_accepted.conf"
BE_VERBOSE=false
###### END CONFIG

if [ "$UID" -ne 0 ]
then
 echo "Superuser rights is required"
 echo 'Printing the # sign'
 exit 2
fi

if test $# -eq 0
then
elif test [ $1 == 'start' ]
else
fi

genApacheConf(){
 if [[ "$2" = "www" ]]
 then
  full_domain=$1
 else
  full_domain=$2.$1
 fi
 host_root="${APACHE_HOME_DIR}$1/$2/$(title)"
 echo -e "# Host $1/$2 :"
}
{{< /codeblock >}}

# Coffeescript

{{< codeblock lang="coffeescript" >}}
 ###
 Some tests
 ###
 class Animal
   constructor: (@name) ->
   move: (meters) -> alert @name + " moved " + meters + "m."

 class Snake extends Animal
   move: ->
     alert 'Slithering...'
     super 5

 number   = 42; opposite = true


 square = (x) -> x * x

 range = [1..2]
 list = [1...5]

 math =
   root:   Math.sqrt
   cube:   (x) => x * square x

 race = (winner, runners...) ->
   print winner, runners

 alert "I knew it!" if elvis?

 cubes = math.cube num for num in list

 text = """
  Result
     is #{ @number }"""

 html = '''   <body></body>'''

 String::dasherize = ->
   this.replace /_/g, "-"
 SINGERS = {Jagger: "Rock", Elvis: "Roll"}

 t = ///
 #{ something }[a-z]
 ///

 $('.shopping_cart').bind 'click', (event) =>
     @customer.purchase @cart

 hi = `function() {
   return [document.title, "Hello JavaScript"].join(": ");
 }`
{{< /codeblock >}}

# C++

{{< codeblock "archives.cpp" "cpp" "http://underscorejs.org/#compact" "archives.cpp" >}}
/*
 * Block comment
 */
#include <vector>

using namespace std;  // line comment
namespace foo {

  typedef struct Struct {
    int field;
  } Typedef;
  enum Enum {Foo = 1, Bar = 2};

  Typedef *globalVar;
  extern Typedef *externVar;

  template<typename T, int N>
  class Class {
    T n;
  public:
    void function(int paramName) {
      int *localVar = new int[1];
      this->n = N;

    label:
      printf("Formatted string %d\n\g", localVar[0]);
      printf(R"**(Formatted raw-string %d\n)**", 1);
      std::cout << (1 << 2) << std::endl;

    #define FOO(A) A
    #ifdef DEBUG
      printf("debug");
    #endif
    }
  };
}
{{< /codeblock >}}

# CShparp

{{< codeblock "archives.cs" "cs" "http://underscorejs.org/#compact" "archives.cs" >}}
using System;

#pragma warning disable 414, 3021

public class Program
{
    /// <summary>The entry point to the program.</summary>
    public static int Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
        string s = @"This
""string""
spans
multiple
lines!";
        return 0;
    }
}

async Task<int> AccessTheWebAsync()
{
    // ...
    string urlContents = await getStringTask;
    return urlContents.Length;
}
{{< /codeblock >}}

# CSS
{{< codeblock "archives.css" "css" "http://underscorejs.org/#compact" "archives.css" >}}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  body:first-of-type pre::after {
    content: 'highlight: ' attr(class);
  }
  body {
    background: linear-gradient(45deg, blue, red);
  }
}

@import url('print.css');
@page:right {
 margin: 1cm 2cm 1.3cm 4cm;
}

@font-face {
  font-family: Chunkfive; src: url('Chunkfive.otf');
}

div.text,
#content,
li[lang=ru] {
  font: Tahoma, Chunkfive, sans-serif;
  background: url('hatch.png') /* wtf? */;  color: #F0F0F0 !important;
  width: 100%;
}
{{< /codeblock >}}

# DIFF
{{< codeblock "archives.diff" "diff" "http://underscorejs.org/#compact" "archives.diff" >}}
Index: languages/ini.js
===================================================================
--- languages/ini.js    (revision 199)
+++ languages/ini.js    (revision 200)
@@ -1,8 +1,7 @@
 hljs.LANGUAGES.ini =
 {
   case_insensitive: true,
-  defaultMode:
-  {
+  defaultMode: {
     contains: ['comment', 'title', 'setting'],
     illegal: '[^\\s]'
   },

*** /path/to/original timestamp
--- /path/to/new      timestamp
***************
*** 1,3 ****
--- 1,9 ----
+ This is an important
+ notice! It should
+ therefore be located at
+ the beginning of this
+ document!

! compress the size of the
! changes.

  It is important to spell
{{< /codeblock >}}

# HTTP
{{< codeblock "archives.http" "http" "http://underscorejs.org/#compact" "archives.http" >}}
POST /task?id=1 HTTP/1.1
Host: example.org
Content-Type: application/json; charset=utf-8
Content-Length: 19

{"status": "ok", "extended": true}
{{< /codeblock >}}

# INI
{{< codeblock "archives.ini" "ini" "http://underscorejs.org/#compact" "archives.ini" >}}
;Settings relating to the location and loading of the database
[Database]
ProfileDir=.
ShowProfileMgr=smart
Profile1_Name[] = "\|/_-=MegaDestoyer=-_\|/"
DefaultProfile=True
AutoCreate = no

[AutoExec]
use-prompt="prompt"
Glob=autoexec_*.ini
AskAboutIgnoredPlugins=0
{{< /codeblock >}}

# Java
{{< codeblock "archives.java" "java" "http://underscorejs.org/#compact" "archives.java" >}}
/* Block comment */
import java.util.Date;
/**
 * Doc comment here for <code>SomeClass</code>
 * @see Math#sin(double)
 */
@Annotation (name=value)
public class SomeClass<T extends Runnable> { // some comment
  private T field = null;
  private double unusedField = 12345.67890;
  private UnknownType anotherString = "Another\nStrin\g";
  public static int staticField = 0;

  public SomeClass(AnInterface param, int[] reassignedParam) {
    int localVar = "IntelliJ"; // Error, incompatible types
    System.out.println(anotherString + toString() + localVar);
    long time = Date.parse("1.2.3"); // Method is deprecated
    int reassignedValue = this.staticField;
    reassignedValue ++;
    field.run();
    new SomeClass() {
      {
        int a = localVar;
      }
    };
    reassignedParam = new ArrayList<String>().toArray(new int[0]);
  }
}
enum AnEnum { CONST1, CONST2 }
interface AnInterface {
  int CONSTANT = 2;
  void method();
}
abstract class SomeAbstractClass {
}
{{< /codeblock >}}

# JavaScript
{{< codeblock "archives.js" "js" "http://underscorejs.org/#compact" "archives.js" >}}
function $initHighlight(block, flags) {
  try {
    if (block.className.search(/\bno\-highlight\b/) != -1)
      return processBlock(block.function, true, 0x0F) + ' class=""';
  } catch (e) {
    /* handle exception */
    var e4x =
        <div>Example
            <p>1234</p></div>;
  }
  for (var i = 0 / 2; i < classes.length; i++) { // "0 / 2" should not be parsed as regexp
    if (checkCondition(classes[i]) === undefined)
      return /\d+[\s/]/g;
  }
  console.log(Array.every(classes, Boolean));
}
{{< /codeblock >}}

# JSON
{{< codeblock "archives.json" "json" "http://underscorejs.org/#compact" "archives.json" >}}
[
  {
    "title": "apples",
    "count": [12000, 20000],
    "description": {"text": "...", "sensitive": false}
  },
  {
    "title": "oranges",
    "count": [17500, null],
    "description": {"text": "...", "sensitive": false}
  }
]
{{< /codeblock >}}

# Makefile

{{< codeblock "archives.mak" "mak" "http://underscorejs.org/#compact" "archives.mak" >}}
# Makefile

BUILDDIR      = _build
EXTRAS       ?= $(BUILDDIR)/extras

.PHONY: main clean

main:
	@echo "Building main facility..."
	build_main $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)/*
{{< /codeblock >}}

# Markdown

{{< codeblock "archives.md" "md" "http://underscorejs.org/#compact" "archives.md" >}}
# hello world

you can write text [with links](http://example.com) inline or [link references][1].

* one _thing_ has *em*phasis
* two __things__ are **bold**

[1]: http://example.com

---

hello world
===========

<this_is inline="xml"></this_is>

> markdown is so cool

    so are code segments

1. one thing (yeah!)
2. two thing `i can write code`, and `more` wipee!
{{< /codeblock >}}

# Nginx

{{< codeblock "archives.conf" "nginx" "http://underscorejs.org/#compact" "archives.conf" >}}
user  www www;
worker_processes  2;
pid /var/run/nginx.pid;
error_log  /var/log/nginx.error_log  debug | info | notice | warn | error | crit;

events {
    connections   2000;
    use kqueue | rtsig | epoll | /dev/poll | select | poll;
}

http {
    log_format main      '$remote_addr - $remote_user [$time_local] '
                         '"$request" $status $bytes_sent '
                         '"$http_referer" "$http_user_agent" '
                         '"$gzip_ratio"';

    send_timeout 3m;
    client_header_buffer_size 1k;

    gzip on;
    gzip_min_length 1100;

    #lingering_time 30;

    server {
        server_name   one.example.com  www.one.example.com;
        access_log   /var/log/nginx.access_log  main;

        rewrite (.*) /index.php?page=$1 break;

        location / {
            proxy_pass         http://127.0.0.1/;
            proxy_redirect     off;
            proxy_set_header   Host             $host;
            proxy_set_header   X-Real-IP        $remote_addr;
            charset            koi8-r;
        }

        location /api/ {
            fastcgi_pass 127.0.0.1:9000;
        }

        location ~* \.(jpg|jpeg|gif)$ {
            root         /spool/www;
        }
    }
{{< /codeblock >}}

# Objective-C

{{< codeblock "archives.m" "objectivec" "http://underscorejs.org/#compact" "archives.m" >}}
#import <UIKit/UIKit.h>
#import "Dependency.h"

@protocol WorldDataSource
@optional
- (NSString*)worldName;
@required
- (BOOL)allowsToLive;
@end

@interface Test : NSObject <HelloDelegate, WorldDataSource> {
  NSString *_greeting;
}

@property (nonatomic, readonly) NSString *greeting;
- (IBAction) show;
@end

@implementation Test

@synthesize test=_test;

+ (id) test {
  return [self testWithGreeting:@"Hello, world!\nFoo bar!"];
}

+ (id) testWithGreeting:(NSString*)greeting {
  return [[[self alloc] initWithGreeting:greeting] autorelease];
}

- (id) initWithGreeting:(NSString*)greeting {
  if ( (self = [super init]) ) {
    _greeting = [greeting retain];
  }
  return self;
}

- (void) dealloc {
  [_greeting release];
  [super dealloc];
}

@end
{{< /codeblock >}}

# Perl

{{< codeblock "archives.perl" "perl" "http://underscorejs.org/#compact" "archives.perl" >}}
# loads object
sub load
{
  my $flds = $c->db_load($id,@_) || do {
    Carp::carp "Can`t load (class: $c, id: $id): '$!'"; return undef
  };
  my $o = $c->_perl_new();
  $id12 = $id / 24 / 3600;
  $o->{'ID'} = $id12 + 123;
  #$o->{'SHCUT'} = $flds->{'SHCUT'};
  my $p = $o->props;
  my $vt;
  $string =~ m/^sought_text$/;
  $items = split //, 'abc';
  $string //= "bar";
  for my $key (keys %$p)
  {
    if(${$vt.'::property'}) {
      $o->{$key . '_real'} = $flds->{$key};
      tie $o->{$key}, 'CMSBuilder::Property', $o, $key;
    }
  }
  $o->save if delete $o->{'_save_after_load'};

  # GH-117
  my $g = glob("/usr/bin/*");

  return $o;
}

=head1 NAME
POD till the end of file
{{< /codeblock >}}

# PHP

{{< codeblock "archives.php" "php" "http://underscorejs.org/#compact" "archives.php" >}}
<?php
$heredoc = <<< HEREDOC_ID
some $contents
HEREDOC_ID;

function foo() {
   return SomeClass::$shared;
}

// Sample comment

class SomeClass extends One implements Another {
   private $my;
   public static $shared;
   const MAGIC = 0987654321;
   /**
    * Description by <a href="mailto:">user@host.dom</a>
    * @return SomeType
    */
   function doSmth($abc, $def) {
      foo();
      $def .=  self::MAGIC;
      $v = Helper::convert($abc . "\n {$def}" . $$def);
      $q = new Query( $this->invent(abs(0x80)) );
      return array($v => $q->result);
   }
}

interface Another {
}

include (dirname(__FILE__) . "inc.php");
`rm -r`;

goto Label;

Label:
<php_bad>№</php_bad>
{{< /codeblock >}}

# Python

{{< codeblock "archives.py" "python" "http://underscorejs.org/#compact" "archives.py" >}}
@requires_authorization
def somefunc(param1='', param2=0):
    r'''A docstring'''
    if param1 > param2: # interesting
        print 'Gre\'ater'
    return (param2 - param1 + 1 + 0b10l) or None

class SomeClass:
    pass

>>> message = '''interpreter
... prompt'''
{{< /codeblock >}}

# Ruby

{{< codeblock "archives.rb" "ruby" "http://underscorejs.org/#compact" "archives.rb" >}}
class A < B; def self.create(object = User) object end end
class Zebra; def inspect; "X#{2 + self.object_id}" end end

module ABC::DEF
  include Comparable

  # @param test
  # @return [String] nothing
  def foo(test)
    Thread.new do |blockvar|
      ABC::DEF.reverse(:a_symbol, :'a symbol', :<=>, 'test' + ?\012)
      answer = valid?4 && valid?CONST && ?A && ?A.ord
    end.join
  end

  def [](index) self[index] end
  def ==(other) other == self end
end

class Car < ActiveRecord::Base
  has_many :wheels, class_name: 'Wheel', foreign_key: 'car_id'
  scope :available, -> { where(available: true) }
end

hash = {1 => 'one', 2 => 'two'}

2.0.0p0 :001 > ['some']
 => ["some"]
{{< /codeblock >}}

# SQL

{{< codeblock "archives.sql" "sql" "http://underscorejs.org/#compact" "archives.sql" >}}
BEGIN;
CREATE TABLE "topic" (
    -- This is the greatest table of all time
    "id" serial NOT NULL PRIMARY KEY,
    "forum_id" integer NOT NULL,
    "subject" varchar(255) NOT NULL -- Because nobody likes an empty subject
);
ALTER TABLE "topic" ADD CONSTRAINT forum_id FOREIGN KEY ("forum_id") REFERENCES "forum" ("id");

-- Initials
insert into "topic" ("forum_id", "subject") values (2, 'D''artagnian');

select /* comment */ count(*) from cicero_forum;

-- this line lacks ; at the end to allow people to be sloppy and omit it in one-liners
/*
but who cares?
*/
COMMIT
{{< /codeblock >}}

# HTML

{{< codeblock "archives.html" "xml" "http://underscorejs.org/#compact" "archives.html" >}}
<?xml version="1.0"?>
<response value="ok" xml:lang="en">
  <text>Ok</text>
  <comment html_allowed="true"/>
  <ns1:description><![CDATA[
  CDATA is <not> magical.
  ]]></ns1:description>
  <a></a> <a/>
</response>


<!DOCTYPE html>
<title>Title</title>

<style>body {width: 500px;}</style>

<script type="application/javascript">
  function $init() {return true;}
</script>

<body>
  <p checked class="title" id='title'>Title</p>
  <!-- here goes the rest of the page -->
<!--</body>
{{< /codeblock >}}

# Puppet

{{< codeblock "archives.pp" "puppet" "http://underscorejs.org/#compact" "archives.pp" >}}
class hg_punch::library {

  firewall {'101 puppet library access':
    proto       => 'tcp',
    dport       => '80',
    action      => 'accept',
  }

  package { 'git':
    ensure => present,
  }

  vcsrepo { "puppet-library":
    path => '/var/www/puppet-library/',
    ensure => present,
    owner => 'root',
    group => 'root',
    provider => git,
    source => 'https://github.com/Moliholy/puppet-library.git',
    revision => 'master',
    require => Package['git'],
  }

  package { 'nfs-utils':
    ensure => present,
  }

  package { 'bundler':
    ensure => present,
    provider => gem,
  }

  package { [ "ruby", "ruby-devel", "gcc", "make" ]:
    ensure => present,
  }

  exec { 'bundler update':
    command => "bundler update && bundler",
    cwd => '/var/www/puppet-library',
    path => ["/usr/bin", "/bin", "/usr/sbin"],
    require => [ Package['ruby'], Package['ruby-devel'],
                Package['gcc'], Package['make'],
                Package['bundler'], Vcsrepo['puppet-library'] ]
  }

  package { 'mod_passenger':
    ensure => present,
  }

  file { "/etc/httpd/conf.d/puppetlibrary.conf":
    owner   => root,
    group   => root,
    mode    => 0644,
    content => template('hg_punch/puppetlibrary.conf.erb'),
    require => Package['mod_passenger'],
    selinux_ignore_defaults => true,
  }

  file { "/var/www/puppet-library/config.ru":
    owner   => root,
    group   => root,
    mode    => 0644,
    content => template('hg_punch/config.ru.erb'),
    require => Vcsrepo['puppet-library'],
  }

  file { [ '/var/www/puppet-library/public', '/var/www/puppet-library/tmp' ]:
    ensure => directory,
    owner   => root,
    group   => root,
    mode => 755,
    require => Vcsrepo['puppet-library'],
  }

  # Disable SELinux
  package { "augeas":
    ensure => present,
  }

  augeas {'disable_selinux':
    context => '/files/etc/sysconfig/selinux',
    changes => 'set SELINUX disabled',
    lens    => 'shellvars.lns',
    incl     => '/etc/sysconfig/selinux'
  } ~>
  exec {'sudo disable_selinux':
    command => '/bin/echo 0 > /selinux/enforce',
    refreshonly => true,
  }

  service { "httpd":
    enable => true,
    ensure => running,
    hasrestart => true,
    require => [ Exec['bundler update'],
                File['/etc/httpd/conf.d/puppetlibrary.conf'],
                File['/var/www/puppet-library/public'],
                File['/var/www/puppet-library/tmp'],
                Vcsrepo['puppet-library'],
                Package['mod_passenger'] ],
  }

}
{{< /codeblock >}}

# Less

{{< codeblock "archives.less" "less" "http://underscorejs.org/#compact" "archives.less" >}}
@import 'mixins'; // external mixins

@the-border: 1px;
@base-color: #111;

#header:after {
  color: @base-color * 3;
  border-right: @the-border * 2;
}

.colored(@c) when (iscolor(@c)) {
  color: (@base-color + #111) * 1.5;
}
@var: `"hello".toUpperCase() + '!'`;

@font-face {
  font-family: DroidSans;
  src: url(DroidSans.ttf);
  unicode-range: U+000-5FF, U+1e00-1fff, U+2000-2300;
}

div > p, p ~ ul, input[type="radio"] {
  color: green !important;
}
{{< /codeblock >}}

# SCSS

{{< codeblock "archives.scss" "scss" "http://underscorejs.org/#compact" "archives.scss" >}}
.btn {
    font-size:      $font-size-base;
    background:     #fff;
    width:          auto;
    height:         auto;
    border-radius:  3px;
    letter-spacing: $letter-spacing-base;
    padding:        8px 15px;
    cursor:         pointer;
    margin:         0;

    &:hover,
    &:focus,
    &:active {
        text-decoration: none;
    }
}

// Colors variant
.btn--default {
    @include button-color-variant($font-color-light);
}
.btn--success {
    @include button-color-variant($color-success);
}
.btn--primary {
    @include button-color-variant($color-primary);
}
.btn--danger {
    @include button-color-variant($color-danger);
}

// Size variant
.btn--medium {
    @include button-size-variant($font-size-medium, 8px, 15px);
}
.btn--small {
    @include button-size-variant($font-size-small, 8px, 15px);
}

// States variant
.btn--disabled,
.btn--disabled:hover {
    color:           lighten($font-color-light, 10) !important;
    border:          1px solid lighten($font-color-light, 10);
    cursor:          not-allowed;
    text-decoration: none;
}
{{< /codeblock >}}

# Stylus

{{< codeblock "archives.styl" "stylus" "http://underscorejs.org/#compact" "archives.styl" >}}
@import "nib"

// variables
$green = #008000
$green_dark = darken($green, 10)

// mixin/function
container()
  max-width 980px

// mixin/function with parameters
buttonBG($color = green)
  if $color == green
    background-color #008000
  else if $color == red
    background-color #B22222

button
  buttonBG(red)

#content, .content
  font Tahoma, Chunkfive, sans-serif
  background url('hatch.png')
  color #F0F0F0 !important
  width 100%
{{< /codeblock >}}

# Go

{{< codeblock "archives.go" "go" "http://underscorejs.org/#compact" "archives.go" >}}
package main

import (
    "fmt"
    "os"
)

const (
    Sunday = iota
    numberOfDays  // this constant is not exported
)

type Foo interface {
    FooFunc(int, float32) (complex128, []int)
}

type Bar struct {
    os.File /* multi-line
               comment */
    PublicData chan int
}

func main() {
    ch := make(chan int)
    ch <- 1
    x, ok := <- ch
    ok = true
    float_var := 1.0e10
    defer fmt.Println('\'')
    defer fmt.Println(`exitting now\`)
    var fv1 float64 = 0.75
    go println(len("hello world!"))
    return
}
{{< /codeblock >}}

# Swift

{{< codeblock "archives.swift" "swift" "http://underscorejs.org/#compact" "archives.swift" >}}
extension MyClass : Interface {
    class func unarchiveFromFile<A>(file : A, (Int,Int) -> B) -> SKNode? {
        let path: String = bundle.pathForResource(file, ofType: "file\(name + 5).txt")
        let funnyNumber = 3 + 0xC2.15p2 * (1_000_000.000_000_1 - 000123.456) + 0o21
        var sceneData = NSData.dataWithContentsOfFile(path, options: .DataReadingMappedIfSafe, error: nil)
        /* a comment /* with a nested comment */ and the end */
    }
    @objc override func shouldAutorotate() {
        return true
    }
}
{{< /codeblock >}}
--->
