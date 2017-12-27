package com.sparkML;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;

import java.util.Arrays;

public class VectorAssemblerExample {
	public static void main(String[] args) {
		vectorAssemblerExample();
	}
	public static void vectorAssemblerExample()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("VectorAssemblerExample").master("local[2]")
			      .getOrCreate();

			    // $example on$
			    StructType schema = createStructType(new StructField[]{
			      createStructField("id", IntegerType, false),
			      createStructField("hour", IntegerType, false),
			      createStructField("mobile", DoubleType, false),
			      createStructField("userFeatures", new VectorUDT(), false),
			      createStructField("clicked", DoubleType, false)
			    });
			    Row row = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0);
			    Dataset<Row> dataset = spark.createDataFrame(Arrays.asList(row), schema);

			    VectorAssembler assembler = new VectorAssembler()
			      .setInputCols(new String[]{"hour", "mobile", "userFeatures"})
			      .setOutputCol("features");

			    Dataset<Row> output = assembler.transform(dataset);
			    System.out.println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column " +
			        "'features'");
			    output.select("features", "clicked").show(false);
			    spark.stop();
	}
}
