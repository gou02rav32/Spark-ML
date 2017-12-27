package com.sparkML;

import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.*;

public class StringIndexerExample {
	public static void main(String[] args) {
		stringIndexer();
	}
	public static void stringIndexer()
	{
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaStringIndexerExample").master("local[2]")
			      .getOrCreate();

			    // $example on$
			    List<Row> data = Arrays.asList(
			      RowFactory.create(0, "a"),
			      RowFactory.create(1, "b"),
			      RowFactory.create(2, "c"),
			      RowFactory.create(3, "a"),
			      RowFactory.create(4, "a"),
			      RowFactory.create(5, "c")
			    );
			    StructType schema = new StructType(new StructField[]{
			      createStructField("id", IntegerType, false),
			      createStructField("category", StringType, false)
			    });
			    Dataset<Row> df = spark.createDataFrame(data, schema);

			    StringIndexer indexer = new StringIndexer()
			      .setInputCol("category")
			      .setOutputCol("categoryIndex");

			    Dataset<Row> indexed = indexer.fit(df).transform(df);
			    indexed.show();
			    spark.stop();
	}
}
