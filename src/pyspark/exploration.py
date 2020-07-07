

import pandas as pd
import numpy as np

import pyspark
import pyspark.sql as sparksql
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf


conf = (SparkConf().setAppName("Kartosiak")
        .set("spark.shuffle.service.enabled", "false")
        .set("spark.dynamicAllocation.enabled", "false")
        .set("spark.io.compression.codec", "snappy")
        # .set("spark.cores.max", "2")
        # .set("spark.rdd.compress", "true")
        .set("spark.executor.instances", "4")
        .set("spark.executor.memory", "512m")
        .set("spark.executor.cores", "5"))

# sc = SparkContext(conf=conf)
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

print(f"INFO: {spark}")


train_df = spark.read.csv(
    "../../data/train_2v.csv",
    inferSchema=True,
    header=True
)

train_df.printSchema()

print(f"[INFO]: COUNT by Group: {train_df.groupBy('stroke').count().show()}")
