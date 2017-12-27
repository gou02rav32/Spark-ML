package com.sparkML;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class PolynominalExpExample {
	public static void main(String[] args) {
		polynominalExp();
	}
	public static void polynominalExp()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaPolynomialExpansionExample").master("local[2]")
			      .getOrCreate();
			    PolynomialExpansion polyExpansion = new PolynomialExpansion()
			      .setInputCol("features")
			      .setOutputCol("polyFeatures")
			      .setDegree(3);
			    List<Row> data = Arrays.asList(
			      RowFactory.create(Vectors.dense(2.0, 1.0)),
			      RowFactory.create(Vectors.dense(0.0, 0.0)),
			      RowFactory.create(Vectors.dense(3.0, -1.0))
			    );
			    StructType schema = new StructType(new StructField[]{
			      new StructField("features", new VectorUDT(), false, Metadata.empty()),
			    });
			    Dataset<Row> df = spark.createDataFrame(data, schema);
			    Dataset<Row> polyDF = polyExpansion.transform(df);
			    polyDF.show(false);
			    spark.stop();
	}
}
