# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:58:48 2015

@author: Abhishek Shivkumar
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

def parsePoint(line):
    return LabeledPoint(line[-1], line[1:-1])


if __name__ == "__main__":
    
    
    conf = SparkConf().setAppName("News Popularity").setMaster("local")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.python.worker.memory", "4g")
    
    sc = SparkContext(conf=conf)
    raw_data = sc.textFile("/home/abhishek/Apache_Spark/exercise/News_Popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv", minPartitions=100)
    
    raw_data_split = raw_data.map(lambda line: [val.strip() for val in line.split(',')])
    
    raw_data_1 = raw_data_split.take(1)
    
    header_value = raw_data_1[0][0]
    
    raw_data_csv = raw_data_split.filter(lambda line: line[0] != header_value)
    
    parsedData = raw_data_csv.map(parsePoint)
    
    (trainingData, testData) = parsedData.randomSplit([0.7, 0.3])
    
    model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=10, maxBins=5)
                                    
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)   
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    
    print('Test Mean Squared Error = ' + str(testMSE))
    print('Learned regression tree model:')
    print(model.toDebugString())
    
    # Save and load model
    model.save(sc, "/home/abhishek/Apache_Spark/exercise/News_Popularity/OnlineNewsPopularity/myModelPath")
    sameModel = DecisionTreeModel.load(sc, "myModelPath")