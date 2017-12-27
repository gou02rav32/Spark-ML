package com.sparkML;

import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class PCAExample {
	public static void main(String[] args) {
		pcaExample();
	}
	public static void pcaExample()
	{
		Logger logger = Logger.getLogger(PCAExample.class);
		SparkConf conf = new SparkConf().setAppName("PCA Example").setMaster("local[2]");
	    SparkContext sc = new SparkContext(conf);
	    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

	    // $example on$
	   /* List<Vector> data = Arrays.asList(
	            Vectors.sparse(5, new int[] {1, 3}, new double[] {1.0, 7.0}),
	            Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
	            Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
	    		Vectors.dense(2, 1),
	    		Vectors.dense(3, 5),
	    		Vectors.dense(4, 3),
	    		Vectors.dense(5, 6),
	    		Vectors.dense(6, 7),
	    		Vectors.dense(7, 8)
	    		
	    );*/
	    String path = "/home/bizruntime/eclipse-workspace/sparkML/src/data/pcaData.txt";
		JavaRDD<String> dataset = jsc.textFile(path);
		JavaRDD<Vector> parsedData = dataset.map(s -> {
		String[] sarray = s.split(",");
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
		values[i] = Double.parseDouble(sarray[i]);
		}
		return Vectors.dense(values);
		});

	   // JavaRDD<Vector> rows = jsc.parallelize(data);

	    // Create a RowMatrix from JavaRDD<Vector>.
	    RowMatrix mat = new RowMatrix(parsedData.rdd());
	    logger.info(mat.computeCovariance());
	    logger.info("Number of cols : "+ mat.numCols());
	    logger.info("Number of Rows : "+ mat.numRows());

	    // Compute the top 4 principal components.
	    // Principal components are stored in a local dense matrix.
	    Matrix pc = mat.computePrincipalComponents(3);
	    logger.info(pc);
	    // Project the rows to the linear space spanned by the top 4 principal components.
	    RowMatrix projected = mat.multiply(pc);
	    logger.info("projected value: "+ projected);
	    Vector[] collectPartitions = (Vector[])projected.rows().collect();
	    System.out.println("Projected vector of principal component:");
	    for (Vector vector : collectPartitions) {
	      System.out.println("\t" + vector);
	    }
	    jsc.stop();
	}
}
