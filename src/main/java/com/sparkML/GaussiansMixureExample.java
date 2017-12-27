package com.sparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class GaussiansMixureExample {
	public static void main(String[] args) {
		gausssiansEx();
	}
	public static void gausssiansEx()
	{
		SparkConf conf = new SparkConf().setAppName("GaussianMixture").setMaster("local[2]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		// Load and parse data
		String path = "src/data/gmm_data.txt";
		JavaRDD<String> data = jsc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(s -> {
		String[] sarray = s.trim().split(" ");
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
		values[i] = Double.parseDouble(sarray[i]);
		}
		return Vectors.dense(values);
		});
		parsedData.cache();
		GaussianMixtureModel gmm = new GaussianMixture().setK(2).run(parsedData.rdd());
		// Save and load GaussianMixtureModel
		gmm.save(jsc.sc(), "target/data/GaussianMixtureModel/gaussiandata");
		GaussianMixtureModel sameModel = GaussianMixtureModel.load(jsc.sc(),
		"target/data/GaussianMixtureModel/gaussiandata");
		// Output the parameters of the mixture model
		for (int j = 0; j < gmm.k(); j++) {
		System.out.printf("weight=%f\nmu=%s\nsigma=\n%s\n",
		gmm.weights()[j], gmm.gaussians()[j].mu(), gmm.gaussians()[j].sigma());
		}
		jsc.stop();
	}
}
