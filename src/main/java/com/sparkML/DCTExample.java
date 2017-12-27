package com.sparkML;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.DCT;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class DCTExample {
	public static void main(String[] args) {
		dctExample();
	}
	public static void dctExample()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaDCTExample").master("local[2]")
			      .getOrCreate();

			    // $example on$
			    List<Row> data = Arrays.asList(
			      RowFactory.create(Vectors.dense(0.0, 1.0, -2.0, 3.0)),
			      RowFactory.create(Vectors.dense(-1.0, 2.0, 4.0, -7.0)),
			      RowFactory.create(Vectors.dense(14.0, -2.0, -5.0, 1.0))
			    );
			    StructType schema = new StructType(new StructField[]{
			      new StructField("features", new VectorUDT(), false, Metadata.empty()),
			    });
			    Dataset<Row> df = spark.createDataFrame(data, schema);

			    DCT dct = new DCT()
			      .setInputCol("features")
			      .setOutputCol("featuresDCT")
			      .setInverse(false);

			    Dataset<Row> dctDf = dct.transform(df);

			    dctDf.select("featuresDCT").show(false);
			    // $example off$

			    spark.stop();
	}
}
