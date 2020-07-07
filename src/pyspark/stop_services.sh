#!/bin/bash

echo "Stopping all Services..."
for pid in $(ps aux | grep spark | awk '{print $2}');
do
  kill -9 $pid;
done
echo "Stopped all Services..."