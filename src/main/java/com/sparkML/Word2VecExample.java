package com.sparkML;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

public class Word2VecExample {
	public static void main(String[] args) {
		word2Vec();
	}
	public static void word2Vec()
	{
		 Logger logger = Logger.getLogger(RF_IDFExample.class);
		 SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaWord2VecExample").master("local[2]")
			      .getOrCreate();
			    // Input data: Each row is a bag of words from a sentence or document.
			    List<Row> data = Arrays.asList(
			      RowFactory.create(Arrays.asList("Hi I heard about Spark".split(" "))),
			      RowFactory.create(Arrays.asList("I wish Java could use case classes".split(" "))),
			      RowFactory.create(Arrays.asList("Logistic regression models are neat".split(" ")))
			    );
			    StructType schema = new StructType(new StructField[]{
			      new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
			    });
			    Dataset<Row> documentDF = spark.createDataFrame(data, schema);
			    // Learn a mapping from words to Vectors.
			    Word2Vec word2Vec = new Word2Vec()
			      .setInputCol("text")
			      .setOutputCol("result")
			      .setVectorSize(3)
			      .setMinCount(0);

			    logger.info("Explanation: " + word2Vec.explainParams());
			    logger.info("Seed: " + word2Vec.getSeed());
			    Word2VecModel model = word2Vec.fit(documentDF);
			    Dataset<Row> result = model.transform(documentDF);
			    
			    for (Row row : result.collectAsList()) {
			      List<String> text = row.getList(0);
			      Vector vector = (Vector) row.get(1);
			      System.out.println("Text: " + text + " => \nVector: " + vector + "\n");
			    }
			    spark.stop();
	}
}
