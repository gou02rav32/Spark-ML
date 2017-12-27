package com.sparkML;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class BisectingKMeansExample {
	public static void main(String[] args) {
		bisectingMean();
	}
	public static void bisectingMean()
	{
		 SparkConf sparkConf = new SparkConf().setAppName("BisectingKMeans").setMaster("local[2]");
		 JavaSparkContext sc = new JavaSparkContext(sparkConf);
		 List<Vector> localData = Arrays.asList(
		 Vectors.dense(0.1, 0.1), Vectors.dense(0.3, 0.3),
		 Vectors.dense(10.1, 10.1), Vectors.dense(10.3, 10.3),
		 Vectors.dense(20.1, 20.1), Vectors.dense(20.3, 20.3),
		 Vectors.dense(30.1, 30.1), Vectors.dense(30.3, 30.3)
		 );
		 JavaRDD<Vector> data = sc.parallelize(localData, 2);
		 BisectingKMeans bkm = new BisectingKMeans()
		 .setK(4);
		 BisectingKMeansModel model = bkm.run(data);
		 System.out.println("Compute Cost: " + model.computeCost(data));
		 Vector[] clusterCenters = model.clusterCenters();
		 for (int i = 0; i < clusterCenters.length; i++) {
		 Vector clusterCenter = clusterCenters[i];
		 System.out.println("Cluster Center " + i + ": " + clusterCenter);
		 }
		 sc.stop();
	}
}
/*
 * Compute Cost: 0.16000000000000458
Cluster Center 0: [0.2,0.2]
Cluster Center 1: [10.2,10.2]
Cluster Center 2: [20.200000000000003,20.200000000000003]
Cluster Center 3: [30.200000000000003,30.200000000000003]*/
