package com.sparkML;

import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class BucketizerExample {
	public static void main(String[] args) {
		bucketizer();
	}
	public static void bucketizer()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("BucketizerExample").master("local[2]")
			      .getOrCreate();

			    // $example on$
			    double[] splits = {Double.NEGATIVE_INFINITY, -0.5, 0.0, 0.5, Double.POSITIVE_INFINITY};

			    List<Row> data = Arrays.asList(
			      RowFactory.create(-999.9),
			      RowFactory.create(-0.5),
			      RowFactory.create(-0.3),
			      RowFactory.create(0.0),
			      RowFactory.create(0.2),
			      RowFactory.create(999.9)
			    );
			    StructType schema = new StructType(new StructField[]{
			      new StructField("features", DataTypes.DoubleType, false, Metadata.empty())
			    });
			    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);

			    Bucketizer bucketizer = new Bucketizer()
			      .setInputCol("features")
			      .setOutputCol("bucketedFeatures")
			      .setSplits(splits);

			    // Transform original data into its bucket index.
			    Dataset<Row> bucketedData = bucketizer.transform(dataFrame);

			    System.out.println("Bucketizer output with " + (bucketizer.getSplits().length-1) + " buckets");
			    bucketedData.show();
			    // $example off$

			    // $example on$
			    // Bucketize multiple columns at one pass.
			    double[][] splitsArray = {
			      {Double.NEGATIVE_INFINITY, -0.5, 0.0, 0.5, Double.POSITIVE_INFINITY},
			      {Double.NEGATIVE_INFINITY, -0.3, 0.0, 0.3, Double.POSITIVE_INFINITY}
			    };

			    List<Row> data2 = Arrays.asList(
			      RowFactory.create(-999.9, -999.9),
			      RowFactory.create(-0.5, -0.2),
			      RowFactory.create(-0.3, -0.1),
			      RowFactory.create(0.0, 0.0),
			      RowFactory.create(0.2, 0.4),
			      RowFactory.create(999.9, 999.9)
			    );
			    StructType schema2 = new StructType(new StructField[]{
			      new StructField("features1", DataTypes.DoubleType, false, Metadata.empty()),
			      new StructField("features2", DataTypes.DoubleType, false, Metadata.empty())
			    });
			    Dataset<Row> dataFrame2 = spark.createDataFrame(data2, schema2);

			    Bucketizer bucketizer2 = new Bucketizer()
			      .setInputCols(new String[] {"features1", "features2"})
			      .setOutputCols(new String[] {"bucketedFeatures1", "bucketedFeatures2"})
			      .setSplitsArray(splitsArray);
			    // Transform original data into its bucket index.
			    Dataset<Row> bucketedData2 = bucketizer2.transform(dataFrame2);

			    System.out.println("Bucketizer output with [" +
			      (bucketizer2.getSplitsArray()[0].length-1) + ", " +
			      (bucketizer2.getSplitsArray()[1].length-1) + "] buckets for each input column");
			    bucketedData2.show();
			    // $example off$

			    spark.stop();
	}
}
