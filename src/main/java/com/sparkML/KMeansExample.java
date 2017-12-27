package com.sparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class KMeansExample {
	public static void main(String[] args) {
		kmeansEx();
	}
	public static void kmeansEx()
	{
		SparkConf conf = new SparkConf().setAppName("KMeansExample").setMaster("local[2]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		// Load and parse data
		String path = "/home/bizruntime/eclipse-workspace/sparkML/src/data/kmeans-data.txt";
		JavaRDD<String> data = jsc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(s -> {
		String[] sarray = s.split(" ");
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
		values[i] = Double.parseDouble(sarray[i]);
		}
		return Vectors.dense(values);
		});
		parsedData.cache();
		// Cluster the data into two classes using KMeans
		int numClusters = 2;
		int numIterations = 20;
		KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);
		System.out.println("Cluster centers:");
		for (Vector center: clusters.clusterCenters()) {
		System.out.println(" " + center);
		}
		double cost = clusters.computeCost(parsedData.rdd());
		System.out.println("Cost: " + cost);
		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double WSSSE = clusters.computeCost(parsedData.rdd());
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		// Save and load model
		clusters.save(jsc.sc(), "/home/bizruntime/eclipse-workspace/sparkML/target/data/kmeans/KMeansModel");
		KMeansModel sameModel = KMeansModel.load(jsc.sc(), "/home/bizruntime/eclipse-workspace/sparkML/target/data/kmeans/KMeansModel");
		jsc.stop();
	}
}
