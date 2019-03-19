# Databricks notebook source
# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

df = spark.table("abillion")

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from abillion

# COMMAND ----------

df.count()