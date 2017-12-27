package com.sparkML;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;

public class ImputerExample {
	public static void main(String[] args) {
		imputerExample();
	}
	public static void imputerExample()
	{
		 SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaImputerExample").master("local[2]")
			      .getOrCreate();
			    List<Row> data = Arrays.asList(
			      RowFactory.create(1.0, Double.NaN),
			      RowFactory.create(2.0, Double.NaN),
			      RowFactory.create(Double.NaN, 3.0),
			      RowFactory.create(4.0, 4.0),
			      RowFactory.create(5.0, 5.0)
			    );
			    StructType schema = new StructType(new StructField[]{
			      createStructField("a", DoubleType, false),
			      createStructField("b", DoubleType, false)
			    });
			    Dataset<Row> df = spark.createDataFrame(data, schema);

			    Imputer imputer = new Imputer()
			      .setInputCols(new String[]{"a", "b"})
			      .setOutputCols(new String[]{"out_a", "out_b"});

			    ImputerModel model = imputer.fit(df);
			    model.transform(df).show();
			    // $example off$

			    spark.stop();
	}
}
