package com.sparkML;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.SparkConf;

public class FP_GrowthExample {
	public static void main(String[] args) {
		fp_growth();
	}
	public static void fp_growth()
	{
		SparkConf conf = new SparkConf().setAppName("FP-growth Example").setMaster("local[2]");
	    JavaSparkContext sc = new JavaSparkContext(conf);

	    // $example on$
	    JavaRDD<String> data = sc.textFile("/home/bizruntime/eclipse-workspace/sparkML/src/data/fp_growth.txt");

	    JavaRDD<List<String>> transactions = data.map(line -> Arrays.asList(line.split(" ")));

	    FPGrowth fpg = new FPGrowth()
	      .setMinSupport(0.2)
	      .setNumPartitions(10);
	    FPGrowthModel<String> model = fpg.run(transactions);

	    for (FPGrowth.FreqItemset<String> itemset: model.freqItemsets().toJavaRDD().collect()) {
	      System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
	    }

	    double minConfidence = 0.8;
	    for (AssociationRules.Rule<String> rule
	      : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
	      System.out.println(
	        rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
	    }
	    // $example off$

	    sc.stop();
	}
}
