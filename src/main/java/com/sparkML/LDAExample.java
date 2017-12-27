package com.sparkML;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

public class LDAExample {
	public static void main(String[] args) {
		ldaExample();
	}
	public static void ldaExample()
	{
		SparkConf conf = new SparkConf().setAppName("Latent Dirichlet Allocation").setMaster("local[2]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		// $example on$
		// Load and parse the data
		String path = "src/data/lda_data.txt";
		JavaRDD<String> data = jsc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(s -> {
		String[] sarray = s.trim().split(" ");
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
		values[i] = Double.parseDouble(sarray[i]);
		}
		return Vectors.dense(values);
		});
		// Index documents with unique IDs
		JavaPairRDD<Long, Vector> corpus =
		JavaPairRDD.fromJavaRDD(parsedData.zipWithIndex().map(Tuple2::swap));
		corpus.cache();
		// Cluster the documents into three topics using LDA
		LDAModel ldaModel = new LDA().setK(3).run(corpus);
		// Output topics. Each is a distribution over words (matching word count vectors)
		System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize()
		+ " words):");
		Matrix topics = ldaModel.topicsMatrix();
		for (int topic = 0; topic < 3; topic++) {
		System.out.print("Topic " + topic + ":");
		for (int word = 0; word < ldaModel.vocabSize(); word++) {
		System.out.print(" " + topics.apply(word, topic));
		}
		System.out.println();
		}
		ldaModel.save(jsc.sc(),
		"target/data/LatentDirichletAllocation/LDAModel");
		DistributedLDAModel sameModel = DistributedLDAModel.load(jsc.sc(),
		"target/data/LatentDirichletAllocation/LDAModel");
		jsc.stop();
	}
}

/*
Topic 0: 6.5756184731133915 12.04972860842974 2.9302609862188196 10.374834703397351 6.715353258725424 5.506834909460169 11.562779465831753 1.7251480645197583 4.126259452319152 12.175355073030389 13.58125354498377
Topic 1: 9.673867447051709 8.751882197331863 4.309880461953176 11.754832133696386 8.55110944339512 6.608590401069977 12.420253713863529 4.005087319558663 2.3745732440987535 6.158458304411274 11.517567016139477
Topic 2: 9.7505140798349 8.198389194238398 4.759858551828005 17.870333162906263 9.733537297879458 9.884574689469854 7.0169668203047175 4.269764615921579 1.4991673035820954 5.666186622558334 7.901179438876753*/