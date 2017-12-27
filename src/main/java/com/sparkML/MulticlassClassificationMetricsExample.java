package com.sparkML;

import scala.Tuple2;
import org.apache.spark.api.java.*;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

public class MulticlassClassificationMetricsExample {
	public static void main(String[] args) {
		multiclassClassificationEx();
	}
	public static void multiclassClassificationEx()
	{
		SparkConf conf = new SparkConf().setAppName("Multi class Classification Metrics Example").setMaster("local[2]");
	    SparkContext sc = new SparkContext(conf);
	    // $example on$
	    String path = "/home/bizruntime/eclipse-workspace/sparkML/src/data/multiClass_Classification.txt";
	    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();

	    // Split initial RDD into two... [60% training data, 40% testing data].
	    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4}, 11L);
	    JavaRDD<LabeledPoint> training = splits[0].cache();
	    JavaRDD<LabeledPoint> test = splits[1];

	    // Run training algorithm to build the model.
	    LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
	      .setNumClasses(3)
	      .run(training.rdd());

	    // Compute raw scores on the test set.
	    JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
	      new Tuple2<>(model.predict(p.features()), p.label()));

	    // Get evaluation metrics.
	    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

	    // Confusion matrix
	    Matrix confusion = metrics.confusionMatrix();
	    System.out.println("Confusion matrix: \n" + confusion);

	    // Overall statistics
	    System.out.println("Accuracy = " + metrics.accuracy());

	    // Stats by labels
	    for (int i = 0; i < metrics.labels().length; i++) {
	      System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
	        metrics.labels()[i]));
	      System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
	        metrics.labels()[i]));
	      System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
	        metrics.labels()[i]));
	    }

	    //Weighted stats
	    System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
	    System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
	    System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
	    System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

	    // Save and load model
	    model.save(sc, "/home/bizruntime/eclipse-workspace/sparkML/target/data/Multiclass_Classification/multiClasss");
	    LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc,
	      "/home/bizruntime/eclipse-workspace/sparkML/target/data/Multiclass_Classification/multiClasss");
	    sc.stop();
	}
}
