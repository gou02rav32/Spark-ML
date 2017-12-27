package com.sparkML;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class StandardScalerExample {
	public static void main(String[] args) {
		standardScalerex();
	}
	public static void standardScalerex()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaStandardScalerExample").master("local[2]")
			      .getOrCreate();

			    // $example on$
			    Dataset<Row> dataFrame =
			      spark.read().format("libsvm").load("/home/bizruntime/eclipse-workspace/sparkML/src/data/lib_svm.txt");

			    StandardScaler scaler = new StandardScaler()
			      .setInputCol("features")
			      .setOutputCol("scaledFeatures")
			      .setWithStd(true)
			      .setWithMean(false);

			    // Compute summary statistics by fitting the StandardScaler
			    StandardScalerModel scalerModel = scaler.fit(dataFrame);

			    // Normalize each feature to have unit standard deviation.
			    Dataset<Row> scaledData = scalerModel.transform(dataFrame);
			    scaledData.show();
			    spark.stop();
	}
}
