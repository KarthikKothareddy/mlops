#!/bin/bash

echo "Starting Spark Master..."
SPARK_HOME="/usr/local/Cellar/apache-spark/3.0.0/libexec"
sh ${SPARK_HOME}/sbin/start-master.sh

echo "Starting Spark Slave..."
SPARK_MASTER="spark://DEVL02Y50DQJGH5:7077"
sh ${SPARK_HOME}/sbin/start-slave.sh ${SPARK_MASTER}

echo "All Services Started..."