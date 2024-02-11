# Databricks notebook source
# MAGIC %md ###Gerekli kütüphaneler import edilir

# COMMAND ----------

pip install nltk

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

# COMMAND ----------

# MAGIC %md ### 1. Veri İndirme

# COMMAND ----------

# MAGIC %md #####Veri kümesini curl kullanarak yerel sürücü düğümünün /tmp klasörüne indirin

# COMMAND ----------

# MAGIC %sh curl -Lb /tmp/gcokie "https://drive.google.com/uc?export=download&id=1nWI-JmZFwhComOPHW8myrtLUjJnC_yvu" -o "/tmp/abcnews-date-text.csv"

# COMMAND ----------

# MAGIC %fs ls file:/tmp/

# COMMAND ----------

# MAGIC %md #####Dosyayı yerel sürücü düğümünün dosya sisteminden DBFS'ye taşıyın

# COMMAND ----------

dbutils.fs.mv("file:/tmp/abcnews-date-text.csv", "dbfs:/datasets/abcnews-date-text.csv")

# COMMAND ----------

# MAGIC %fs ls /datasets/

# COMMAND ----------

# MAGIC %md #####Veri kümesi dosyasını bir Spark Dataframe'i olarak okuyun

# COMMAND ----------

news_df = spark.read.load("dbfs:/datasets/abcnews-date-text.csv", 
                         format="csv", 
                         sep=",", 
                         inferSchema="true", 
                         header="true"
                         )

# COMMAND ----------

# MAGIC %md #####Yüklenen veri kümesinin satır ve sütun sayısını kontrol edin

# COMMAND ----------

print("The shape of the dataset is {:d} rows by {:d} columns".format(news_df.count(), len(news_df.columns)))

# COMMAND ----------

# MAGIC %md #####Yüklenen veri kümesinin şemasını yazdırın

# COMMAND ----------

news_df.printSchema()

# COMMAND ----------

# MAGIC %md #####Veri kümesinin ilk 5 satırını görüntüleme

# COMMAND ----------

news_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ##### Yinelenen haberlerin sayısını (varsa) hesaplama

# COMMAND ----------

print("The total number of duplicated news are {:d} out of {:d}".
      format(news_df.count() - news_df.dropDuplicates(['headline_text']).count(), news_df.count()))

# COMMAND ----------

# MAGIC %md ##### En çok tekrar eden ilk 10 haberi görüntüleyin

# COMMAND ----------

news_df.groupby(["headline_text"]).count().sort("count", ascending=False).show(10)

# COMMAND ----------

# MAGIC %md ##### Yinelenen haberleri kaldırma

# COMMAND ----------

news_df = news_df.dropDuplicates(["headline_text"])

# COMMAND ----------

print("The total number of unique news is: {:d}".format(news_df.count()))

# COMMAND ----------

# MAGIC %md ##### Headline_text sütunu boyunca herhangi bir eksik değer (NULL) olup olmadığını kontrol etme

# COMMAND ----------

news_df.where(col("headline_text").isNull()).count()
# Alternatively, using filter:
# news_df.filter(news_df.headline_text.isNull()).count()

# COMMAND ----------

# MAGIC %md ### 2. Veri Ön İşleme

# COMMAND ----------

# MAGIC %md Bu örnekte, metin verileriyle çalışıyoruz ve nihai hedefimiz, bildiğimiz kümeleme algoritmalarından birini (örneğin, K-means) kullanarak haberleri tutarlı "konu" gruplarına ayırmaktır. Bu, doğal dil işleme (NLP) olarak adlandırılan daha genel bir alanın özel bir problemidir.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###### Herhangi bir NLP görevinin ön adımları olarak, önce en azından aşağıdaki ardışık düzen yürütülmelidir:
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC - Metin temizleme:
# MAGIC  - Büyük/küçük harf normalleştirme (<code>lower</code>) -> tüm metni küçük harfe çevir;
# MAGIC  - Baştaki ve sondaki boşluklarını filtreleyin (<code>trim</code>);
# MAGIC  - Noktalama işaretlerini filtreleyin (<code>regexp_replace</code>);
# MAGIC  - Yukarıdaki adımdan kaynaklanan dahili fazladan boşlukları filtreleyin (<code>regexp_replace</code> + <code>trim</code>).
# MAGIC - Belirteçleştirme (<code>Tokenizer</code>): genellikle sınırlayıcı olarak boşluk kullanarak, ham metni tek tek _belirteçler_ (yani kelimeler) listesine bölme 
# MAGIC - Gereksiz kelimelerin kaldırılması (<code>StopWordsRemover</code>): _stopwords_ denilen, yani ingilizcede "the", "a", "me" gibi metnin derin anlamına katkıda bulunmayan kelimelerin kaldırılması.
# MAGIC - Stemming (<code>SnowballStemmer</code>): Her kelimeyi köküne veya tabanına indirgeme. Örneğin, "balıkçılık", "balıklar", "balıkçı", hepsi "balık" köküne indirgenir.

# COMMAND ----------

def clean_text(df, column_name="headline_text"):
    """ 
    This function takes the raw text data and applies a standard NLP preprocessing pipeline consisting of the following steps:
      - Text cleaning
      - Tokenization
      - Stopwords removal
      - Stemming (Snowball stemmer)

    parameter: dataframe
    returns: the input dataframe along with the `cleaned_content` column as the results of the NLP preprocessing pipeline

    """
    from pyspark.sql.functions import udf, col, lower, trim, regexp_replace
    from pyspark.ml.feature import Tokenizer, StopWordsRemover
    from nltk.stem.snowball import SnowballStemmer # BE SURE NLTK IS INSTALLED ON THE CLUSTER USING THE "LIBRARIES" TAB IN THE MENU

    # Text preprocessing pipeline
    print("***** Text Preprocessing Pipeline *****\n")

    # 1. Text cleaning
    print("# 1. Text Cleaning\n")
    # 1.a Case normalization
    print("1.a Case normalization:")
    lower_case_news_df = df.select(lower(col(column_name)).alias(column_name))
    lower_case_news_df.show(10)
    # 1.b Trimming
    print("1.b Trimming:")
    trimmed_news_df = lower_case_news_df.select(trim(col(column_name)).alias(column_name))
    trimmed_news_df.show(10)
    # 1.c Filter out punctuation symbols
    print("1.c Filter out punctuation:")
    no_punct_news_df = trimmed_news_df.select((regexp_replace(col(column_name), "[^a-zA-Z\\s]", "")).alias(column_name))
    no_punct_news_df.show(10)
    # 1.d Filter out any internal extra whitespace
    print("1.d Filter out extra whitespaces:")
    cleaned_news_df = no_punct_news_df.select(trim(regexp_replace(col(column_name), " +", " ")).alias(column_name))
    cleaned_news_df.show(10)

    # 2. Tokenization (split text into tokens)
    print("# 2. Tokenization:")
    tokenizer = Tokenizer(inputCol=column_name, outputCol="tokens")
    tokens_df = tokenizer.transform(cleaned_news_df).cache()
    tokens_df.show(10)

    # 3. Stopwords removal
    print("# 3. Stopwords removal:")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="terms")
    terms_df = stopwords_remover.transform(tokens_df).cache()
    terms_df.show(10)

    # 4. Stemming (Snowball stemmer)
    print("# 4. Stemming:")
    stemmer = SnowballStemmer(language="english")
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
    terms_stemmed_df = terms_df.withColumn("terms_stemmed", stemmer_udf("terms")).cache()
    terms_stemmed_df.show(10)
    
    return terms_stemmed_df

# COMMAND ----------

clean_news_df = clean_text(news_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### **Feature Engineering**
# MAGIC 
# MAGIC Makine öğrenimi teknikleri doğrudan metin verileri üzerinde çalışamaz; Aslında, kelimeler önce makine öğrenimi algoritmalarının kullanabileceği bazı sayısal temsillere dönüştürülmelidir. Bu süreç genellikle _vektörleştirme_ olarak bilinir.
# MAGIC 
# MAGIC Vektörleştirme açısından, bunun yalnızca tek bir kelimeyi tek bir sayıya dönüştürmek olmadığını hatırlamak önemlidir. Kelimeler sayılara dönüştürülebilirken, bir belgenin tamamı bir vektöre çevrilebilir. Ayrıca, metin verilerinden türetilen vektörler genellikle yüksek boyutludur. Bunun nedeni, özellik uzayının her boyutunun bir kelimeye karşılık gelmesi ve belgelerdeki dilin binlerce kelimeye sahip olabilmesidir.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## TF-IDF
# MAGIC Bilgi almada, **tf-idf** - terimin kısaltması frekans-ters belge frekansı - bir kelimenin bir koleksiyondaki veya tümcedeki bir belge için ne kadar önemli olduğunu yansıtması amaçlanan sayısal bir istatistiktir.
# MAGIC 
# MAGIC tf-idf değeri, bir sözcüğün belgede görünme sayısıyla orantılı olarak artar ve bazı sözcüklerin genel olarak daha sık göründüğü gerçeğini düzeltmeye yardımcı olan, tümcedeki sözcüğün sıklığıyla dengelenir.

# COMMAND ----------

RANDOM_SEED = 42 # used below to run the actual K-means clustering
VOCAB_SIZE = 1000 # number of words to be retained as vocabulary
MIN_DOC_FREQ = 10 # minimum number of documents a word has to appear in to be included in the vocabulary
N_GRAMS = 2 # number of n-grams (if needed)
N_FEATURES = 200 # default embedding vector size (if HashingTF or, later, Word2Vec are used)

# COMMAND ----------

def extract_tfidf_features(df, column_name="terms_stemmed"):
    """ 
    This fucntion takes the text data and converts it into a term frequency-inverse document frequency vector

    parameter: dataframe
    returns: dataframe with tf-idf vectors

    """

    # Importing the feature transformation classes for doing TF-IDF 
    from pyspark.ml.feature import HashingTF, CountVectorizer, IDF, NGram
    from pyspark.ml import Pipeline

    ## Extracting n-grams from text
    #ngrams = NGram(n=N_GRAMS, inputCol=column_name, outputCol="ngrams")
    #ngrams.transform(df)
    
    ## Creating Term Frequency Vector for each word
    #cv = CountVectorizer(inputCol=column_name, outputCol="tf_features", vocabSize=VOCAB_SIZE, minDF=MIN_DOC_FREQ)
    #cv_model = cv.fit(df)
    #tf_features_df = cv_model.transform(df).cache()

    ## Alternatively to CountVectorizer, use HashingTF
    #hashing_TF = HashingTF(inputCol=column_name, outputCol="tf_features", numFeatures=N_FEATURES)
    #tf_features_df = hashing_TF.transform(df).cache()

    ## Carrying out Inverse Document Frequency on the TF data
    #idf = IDF(inputCol="tf_features", outputCol="features")
    #idf_model = idf.fit(tf_features_df)
    #tf_idf_features_df = idf_model.transform(tf_features_df).cache()

    # USING PIPELINE
    #ngrams = NGram(n=N_GRAMS, inputCol=column_name, outputCol="ngrams")
    cv = CountVectorizer(inputCol=column_name, outputCol="tf_features", vocabSize=VOCAB_SIZE, minDF=MIN_DOC_FREQ)
    # hashingTF = HashingTF(inputCol=column_name, outputCol="tf_features", numFeatures=N_FEATURES)
    idf = IDF(inputCol="tf_features", outputCol="features")

    pipeline = Pipeline(stages=[cv, idf]) # add `ngrams` and replace `cv` with `hashingTF`, if needed
    features = pipeline.fit(df)
    tf_idf_features_df = features.transform(df).cache()

    return tf_idf_features_df

# COMMAND ----------

features = extract_tfidf_features(clean_news_df)

# COMMAND ----------

features.select(col("features")).show(10, truncate=False)

# COMMAND ----------

clean_news_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ###### Olası sıfır uzunluklu vektörleri kontrol edin ve kaldırın

# COMMAND ----------

@udf("long")
def num_nonzeros(v):
    return v.numNonzeros()

# COMMAND ----------

print("Total n. of zero-length vectors: {:d}".
      format(features.where(num_nonzeros("features") == 0).count()))

# COMMAND ----------

features = features.where(num_nonzeros("features") > 0)

# COMMAND ----------

print("Total n. of zero-length vectors (after removal): {:d}".
      format(features.where(num_nonzeros("features") == 0).count()))

# COMMAND ----------

# MAGIC %md ### 3. K-means Clustering

# COMMAND ----------

N_CLUSTERS = 10 # number of output clusters (K)
DISTANCE_MEASURE = "euclidean" # alternatively, "cosine"
MAX_ITERATIONS = 100 # maximum number of iterations of K-means EM algorithm
TOLERANCE = 0.000001 # tolerance between consecutive centroid updates (i.e., another stopping criterion)

# COMMAND ----------

def k_means(dataset, 
            n_clusters, 
            distance_measure=DISTANCE_MEASURE, 
            max_iter=MAX_ITERATIONS, 
            tol=TOLERANCE,
            features_col="features", 
            prediction_col="cluster", 
            random_seed=RANDOM_SEED):
  
  from pyspark.ml.clustering import KMeans

  print("""Training K-means clustering using the following parameters: 
  - K (n. of clusters) = {:d}
  - max_iter (max n. of iterations) = {:d}
  - distance measure = {:s}
  - random seed = {:d}
  """.format(n_clusters, max_iter, distance_measure, random_seed))
  # Train a K-means model
  kmeans = KMeans(featuresCol=features_col, 
                   predictionCol=prediction_col, 
                   k=n_clusters, 
                   initMode="k-means||", 
                   initSteps=5, 
                   tol=tol, 
                   maxIter=max_iter, 
                   seed=random_seed, 
                   distanceMeasure=distance_measure)
  model = kmeans.fit(dataset)

  # Make clusters
  clusters_df = model.transform(dataset).cache()

  return model, clusters_df

# COMMAND ----------

model, clusters_df = k_means(features, N_CLUSTERS, max_iter=MAX_ITERATIONS, distance_measure=DISTANCE_MEASURE)

# COMMAND ----------

# MAGIC %md ##### Elde edilen kümeleri değerlendirmek için kullanılan fonksiyon (Silhouette Coefficient)

# COMMAND ----------



def evaluate_k_means(clusters, 
                     metric_name="silhouette", 
                     distance_measure="squaredEuclidean", # cosine
                     prediction_col="cluster"
                     ):
  
  from pyspark.ml.evaluation import ClusteringEvaluator
  
  # Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator(metricName=metric_name,
                                  distanceMeasure=distance_measure, 
                                  predictionCol=prediction_col
                                  )

  return evaluator.evaluate(clusters)



# COMMAND ----------

# MAGIC %md ######Kümelemeyi değerlendirin (Silhouette Coefficient)

# COMMAND ----------

evaluate_k_means(clusters_df, distance_measure="squaredEuclidean")

# COMMAND ----------

clusters_df.show(5)

# COMMAND ----------

clusters_df.groupBy("cluster").count().sort("cluster").show()

# COMMAND ----------

# Get unique values in the grouping column
clusters = sorted([x[0] for x in clusters_df.select("cluster").distinct().collect()])
print("Cluster IDs: [{:s}]".format(", ".join([str(c) for c in clusters])))

# Create a filtered DataFrame for each group in a list comprehension
cluster_list = [clusters_df.where(clusters_df.cluster == x) for x in clusters]

# Show the results (first 5 cluters)
for x_id, x in enumerate(cluster_list):
  print("Showing the first 10 records of cluster ID #{:d}".format(x_id))
  x.select(["cluster", "headline_text"]).show(10, truncate=True)

# COMMAND ----------



# COMMAND ----------

EMBEDDING_SIZE = 150 # size of embedding Word2Vec vectors

# COMMAND ----------

def extract_w2v_features(df, column_name="terms"):
  from pyspark.ml.feature import Word2Vec
  
  word2vec = Word2Vec(vectorSize=EMBEDDING_SIZE, minCount=5, inputCol=column_name, outputCol="features", seed=RANDOM_SEED)
  model = word2vec.fit(df)
  features = model.transform(df).cache()
  
  return model, features

# COMMAND ----------

model, w2v_features = extract_w2v_features(clean_news_df)

# COMMAND ----------

w2v_features.show(truncate=False)

# COMMAND ----------

vecs = model.getVectors()
syms = model.findSynonyms("doctor", 2)
syms.show()

# COMMAND ----------



# COMMAND ----------

K_MIN = 2
K_MAX = 18
STEP = 2

# COMMAND ----------

def elbow_method(data, k_min=K_MIN, k_max=K_MAX, step=STEP, max_iter=MAX_ITERATIONS, distance_measure=DISTANCE_MEASURE):
  results = []
  for k in range(k_min, k_max, step):
    model, clusters_df = k_means(data, k, max_iter=max_iter, distance_measure=distance_measure)
    results.append([k, model.summary.trainingCost])

  return pd.DataFrame(results, columns = ['K', 'SSE'])

# COMMAND ----------

# Get results from elbow method
elbow_results = elbow_method(w2v_features)
