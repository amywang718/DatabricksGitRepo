# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Evaluating Risk for Loan Approvals
# MAGIC 
# MAGIC ## Business Value
# MAGIC 
# MAGIC Being able to accurately assess the risk of a loan application can save a lender the cost of holding too many risky assets. Rather than a credit score or credit history which tracks how reliable borrowers are, we will generate a score of how profitable a loan will be compared to other loans in the past. The combination of credit scores, credit history, and profitability score will help increase the bottom line for financial institution.
# MAGIC 
# MAGIC Having a interporable model that an loan officer can use before performing a full underwriting can provide immediate estimate and response for the borrower and a informative view for the lender.
# MAGIC 
# MAGIC <a href="https://ibb.co/cuQYr6"><img src="https://preview.ibb.co/jNxPym/Image.png" alt="Image" border="0"></a>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data
# MAGIC 
# MAGIC The data used is public data from Lending Club. It includes all funded loans from 2012 to 2017. Each loan includes applicant information provided by the applicant as well as the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. For a full view of the data please view the data dictionary available [here](https://resources.lendingclub.com/LCDataDictionary.xlsx).
# MAGIC 
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Source library from github

# COMMAND ----------

# MAGIC %sh cd /dbfs/amy/lib/DatabricksGitRepo; git pull

# COMMAND ----------

import sys

libpath = "/dbfs/amy/lib/DatabricksGitRepo/lib"
sys.path.insert(0, libpath)

# COMMAND ----------

import tabulate as tb

table = [["Sun",696000,1989100000],["Earth",6371,5973.6],
         ["Moon",1737,73.5],["Mars",3390,641.85]]
print(tb.tabulate(table))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up streaming data source

# COMMAND ----------

loan_stats = spark.table("amy.loanstats_2012_2017")
schema = loan_stats.schema

loan_stats.write.format("delta").save("/delta/loanstats/")

# COMMAND ----------

streaming_df = spark.readStream\
  .option("maxFilesPerTrigger", 10)\
  .format("delta")\
  .load("/delta/loanstats/")\
  .writeStream\
  .partitionBy("issue_d")\
  .format("delta")\
  .outputMode("append")\
  .option("checkpointLocation", "/delta/loanstats_table/_checkpoint_directory")\
  .start("/delta/loanstats_table") 

# COMMAND ----------

# MAGIC %fs ls /delta/loanstats_table

# COMMAND ----------

# %sql
# use amy;
# create table loans
# using delta
# location "/delta/loanstats_table"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Analysis on Loans Data

# COMMAND ----------

loan_stats = spark.table("amy.loans")
# loan_stats = spark.table("amy.loanstats_2012_2017")
print(str(loan_stats.count()) + " loans opened by Lending Club...")

# COMMAND ----------

display(loan_stats)

# COMMAND ----------

# MAGIC %sql
# MAGIC describe formatted amy.loans

# COMMAND ----------

display(loan_stats)

# COMMAND ----------

from pyspark.sql.functions import *

print("------------------------------------------------------------------------------------------------")
print("Create bad loan label, this will include charged off, defaulted, and late repayments on loans...")
loan_stats = loan_stats.filter(loan_stats.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
                       .withColumn("bad_loan", (~(loan_stats.loan_status == "Fully Paid")).cast("string"))

print("------------------------------------------------------------------------------------------------")
print("Turning string interest rate and revoling util columns into numeric columns...")
loan_stats = loan_stats.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float')) \
                       .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float')) \
                       .withColumn('issue_year',  substring(loan_stats.issue_d, 5, 4).cast('double') ) \
                       .withColumn('earliest_year', substring(loan_stats.earliest_cr_line, 5, 4).cast('double'))
loan_stats = loan_stats.withColumn('credit_length_in_years', (loan_stats.issue_year - loan_stats.earliest_year))


print("------------------------------------------------------------------------------------------------")
print("Converting emp_length column into numeric...")
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "") ))
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "< 1", "0") ))
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(loan_stats.emp_length, "10\\+", "10") ).cast('float'))

print("------------------------------------------------------------------------------------------------")
print("Map multiple levels into one factor level for verification_status...")
loan_stats = loan_stats.withColumn('verification_status', trim(regexp_replace(loan_stats.verification_status, 'Source Verified', 'Verified')))

print("------------------------------------------------------------------------------------------------")
print("Calculate the total amount of money earned or lost per loan...")
loan_stats = loan_stats.withColumn('net', round( loan_stats.total_pymnt - loan_stats.loan_amnt, 2))

# COMMAND ----------

display(loan_stats)

# COMMAND ----------

display(loan_stats.groupBy("addr_state").agg((count(col("annual_inc"))).alias("ratio")))

# COMMAND ----------

display(loan_stats)
# display(loan_stats.groupBy("bad_loan", "grade").agg((sum(col("net"))).alias("sum_net")))

# COMMAND ----------

display(loan_stats.select("net","verification_status","int_rate", "revol_util", "issue_year", "earliest_year", "bad_loan", "credit_length_in_years", "emp_length"))

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict bad loans")
myY = "bad_loan"

categoricals = ["term", "home_ownership", "purpose", "addr_state",
                "verification_status","application_type"]
numerics = ["loan_amnt","emp_length", "annual_inc","dti",
            "delinq_2yrs","revol_util","total_acc",
            "credit_length_in_years"]
myX = categoricals + numerics

loan_stats2 = loan_stats.select(myX + [myY, "int_rate", "net", "issue_year"])
train = loan_stats2.filter(loan_stats2.issue_year <= 2015).cache()
valid = loan_stats2.filter(loan_stats2.issue_year > 2015).cache()

train.registerTempTable("train")
valid.registerTempTable("valid")
# train.count()
# valid.count()

# COMMAND ----------

# valid.write.saveAsTable("amy.scoring_table", format = "parquet", path = "dbfs:/amy/loans_valid", mode = "overwrite")

# COMMAND ----------

# MAGIC %scala
# MAGIC // import ml.dmlc.xgboost4j.scala.spark.{XGBoost,XGBoostEstimator}
# MAGIC // import org.apache.spark.ml.feature._
# MAGIC // import org.apache.spark.ml._
# MAGIC 
# MAGIC // val trainSc = table("train").cache()
# MAGIC // val validSc = table("valid").cache()
# MAGIC 
# MAGIC // val categoricals = Array("term", "home_ownership", "purpose", "addr_state",
# MAGIC //                 "verification_status","application_type")
# MAGIC // val numerics = Array("loan_amnt","emp_length", "annual_inc","dti",
# MAGIC //             "delinq_2yrs","revol_util","total_acc",
# MAGIC //             "credit_length_in_years")
# MAGIC 
# MAGIC 
# MAGIC // val indexers = categoricals.map(x=> new StringIndexer().setInputCol(x).setOutputCol(x + "_idx").setHandleInvalid("keep"))
# MAGIC // val oneHotEncoders = categoricals.map(x=> new OneHotEncoder().setInputCol(x + "_idx").setOutputCol(x + "_class"))
# MAGIC // val imputers = new Imputer().setInputCols(numerics).setOutputCols(numerics)
# MAGIC // val featureCols = categoricals.map(c =>c + "_class").toArray ++ numerics
# MAGIC 
# MAGIC // val assembler =  new VectorAssembler()
# MAGIC //   .setInputCols(featureCols)
# MAGIC //   .setOutputCol("features")
# MAGIC // val scaler = new StandardScaler().setInputCol("features").setOutputCol("features").setWithMean(true).setWithStd(true)
# MAGIC // val labelIndexer = new StringIndexer().setInputCol("bad_loan").setOutputCol("label")
# MAGIC // val pipelineAry = indexers ++ oneHotEncoders ++ Array(imputers,assembler,scaler,labelIndexer)
# MAGIC 
# MAGIC // val xgboostEstimator = new XGBoostEstimator(
# MAGIC //         Map[String, Any]("num_round" -> 5, "objective" -> "binary:logistic", "nworkers" -> 16,"nthreads" -> 4))
# MAGIC // val xgBoostModel =  new Pipeline().setStages(pipelineAry ++ Array(xgboostEstimator)).fit(trainSc)

# COMMAND ----------

# MAGIC %scala
# MAGIC // val validSc = table("valid").cache()
# MAGIC // display(xgBoostModel.transform(validSc))

# COMMAND ----------

# MAGIC %scala
# MAGIC // xgBoostModel.save("dbfs:/amy/xgboost_model")

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
# numerics = []
# categoricals = []
labelCol = "bad_loan"
indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals))
ohes = list(map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"), categoricals))
imputers = Imputer(inputCols = numerics, outputCols = numerics)
featureCols = list(map(lambda c: c+"_class", categoricals) ) + numerics
model_matrix_stages = indexers + ohes + \
                      [imputers] + \
                      [VectorAssembler(inputCols=featureCols, outputCol="features"), \
                       StringIndexer(inputCol= labelCol, outputCol="label")]

scaler = StandardScaler(inputCol="features",
                        outputCol="scaledFeatures",
                        withStd=True,
                        withMean=True)

lr = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "scaledFeatures")

pipeline = Pipeline(stages=model_matrix_stages+[scaler]+[lr])


paramGrid = ParamGridBuilder() \
              .addGrid(lr.regParam, [0.1, 0.01]) \
              .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5)

cvModel = crossval.fit(train)
glm_model = cvModel.bestModel

lr_summary = glm_model.stages[len(glm_model.stages)-1].summary
display(lr_summary.roc)

# COMMAND ----------

fMeasure = lr_summary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
maxFMeasure = maxFMeasure['max(F-Measure)']
fMeasure = fMeasure.toPandas()
bestThreshold = float ( fMeasure[ fMeasure['F-Measure'] == maxFMeasure] ["threshold"])
lr.setThreshold(bestThreshold)

# COMMAND ----------

# MAGIC %md ### Other model builds

# COMMAND ----------

# from pyspark.ml.classification import GBTClassifier

# indexers = list(map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals))
# imputers = Imputer(inputCols = numerics, outputCols = numerics)
# featureCols = list(map(lambda c: c+"_class", categoricals) ) + numerics

# # Define vector assemblers
# model_matrix_stages = indexers + \
#                       [imputers] + \
#                       [VectorAssembler(inputCols=featureCols, outputCol="features"), \
#                        StringIndexer(inputCol="bad_loan", outputCol="label")]

# # Define a GBT model.
# gbt = GBTClassifier(featuresCol="features",
#                     labelCol="label",
#                     lossType = "logistic",
#                     maxBins = 52,
#                     maxIter=20,
#                     maxDepth=5)

# # Chain indexer and GBT in a Pipeline
# pipeline = Pipeline(stages=model_matrix_stages+[gbt])

# # Train model.  This also runs the indexer.
# gbt_model = pipeline.fit(train)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.linalg import Vectors

def extract(row):
  return (row.net,) + tuple(row.probability.toArray().tolist()) +  (row.label,) + (row.prediction,)

def score(model,data):
  pred = model.transform(data).select("net", "probability", "label", "prediction")
  pred = pred.rdd.map(extract).toDF(["net", "p0", "p1", "label", "prediction"])
  return pred 

def auc(pred):
  metric = BinaryClassificationMetrics(pred.select("p1", "label").rdd)
  return metric.areaUnderROC

glm_train = score(glm_model, train)
glm_valid = score(glm_model, valid)
gbt_train = score(gbt_model, train)
gbt_valid = score(gbt_model, valid)

glm_train.registerTempTable("glm_train")
glm_valid.registerTempTable("glm_valid")
gbt_train.registerTempTable("gbt_train")
gbt_valid.registerTempTable("gbt_valid")


print "GLM Training AUC :" + str( auc(glm_train))
print "GLM Validation AUC :" + str(auc(glm_valid))
print "GBT Training AUC :" + str(auc(gbt_train))
print "GBT Validation AUC :" + str(auc(gbt_valid))

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
# MAGIC // import org.apache.spark.sql.functions.typedLit
# MAGIC import org.apache.spark.sql.functions.{array, lit, map, struct}
# MAGIC 
# MAGIC def roc(pred:org.apache.spark.sql.DataFrame, model_id:String): org.apache.spark.sql.DataFrame = {
# MAGIC   var testScoreAndLabel = pred.select("p1", "label").map{ case Row(p:Double,l:Double) => (p,l)}
# MAGIC   val metrics = new BinaryClassificationMetrics(testScoreAndLabel.rdd, 100)
# MAGIC   val roc = metrics.roc().toDF().withColumn("model", lit(model_id))
# MAGIC   return roc
# MAGIC }
# MAGIC 
# MAGIC val glm_train = roc( spark.table("glm_train"), "glm_train")
# MAGIC val glm_valid = roc( spark.table("glm_valid"), "glm_valid")
# MAGIC val gbt_train = roc( spark.table("gbt_train"), "gbt_train")
# MAGIC val gbt_valid = roc( spark.table("gbt_valid"), "gbt_valid")
# MAGIC 
# MAGIC val roc_curves = glm_train.union(glm_valid).union(gbt_train).union(gbt_valid)
# MAGIC 
# MAGIC display(roc_curves)

# COMMAND ----------

display(glm_valid.groupBy("label", "prediction").agg((sum(col("net"))).alias("sum_net")))

# COMMAND ----------

cvModel.bestModel.save("dbfs:/amy/lrModel")

# COMMAND ----------

# from dbmlModelExport import ModelExport
# ModelExport.exportModel(cvModel.bestModel, "dbfs:/amy/myPipeline")

# COMMAND ----------

# %fs ls dbfs:/amy/myPipeline/

# COMMAND ----------

# %sh
# zip -r /dbfs/amy/myPipeline.zip /dbfs/amy/myPipeline/*

# COMMAND ----------

# display(loan_stats)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
kmeans = KMeans().setK(3).setSeed(1)
pipeline = Pipeline(stages=model_matrix_stages+[scaler]+[kmeans])

model = pipeline.fit(valid)

# # Make predictions
# predictions = model.transform(dataset)

# # Evaluate clustering by computing Silhouette score
# evaluator = ClusteringEvaluator()

# silhouette = evaluator.evaluate(predictions)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

# # Shows the result.
# centers = model.clusterCenters()
# print("Cluster Centers: ")
# for center in centers:
#     print(center)


# COMMAND ----------

model.save("dbfs:/amy/cluster_model")