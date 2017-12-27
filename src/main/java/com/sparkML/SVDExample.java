package com.sparkML;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class SVDExample {
	public static void main(String[] args) {
		svdExample();
	}
	public static void svdExample()
	{
		SparkConf conf = new SparkConf().setAppName("SVD Example").setMaster("local[2]");
		SparkContext sc = new SparkContext(conf);
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);
		// $example on$
		List<Vector> data = Arrays.asList(
		/*Vectors.sparse(5, new int[] {1, 3}, new double[] {1.0, 7.0}),
		Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
		Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)*/
		Vectors.dense(1, 1, 1, 0, 0),
		Vectors.dense(3, 3, 3, 0, 0),
		Vectors.dense(4, 4, 4, 0, 0),
		Vectors.dense(5, 5, 5, 0, 0),
		Vectors.dense(0, 2, 0, 4, 4),
		Vectors.dense(0, 0, 0, 5, 5),
		Vectors.dense(0, 1, 0, 2, 2)
		);
		JavaRDD<Vector> rows = jsc.parallelize(data);
		// Create a RowMatrix from JavaRDD<Vector>.
		RowMatrix mat = new RowMatrix(rows.rdd());
		// Compute the top 5 singular values and corresponding singular vectors.
		SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(5, true, 1.0E-9d);
		RowMatrix U = svd.U(); // The U factor is a RowMatrix.
		Vector s = svd.s(); // The singular values are stored in a local dense vector.
		Matrix V = svd.V(); // The V factor is a local dense matrix.
		// $example off$
		Vector[] collectPartitions = (Vector[]) U.rows().collect();
		System.out.println("U factor is:");
		for (Vector vector : collectPartitions) {
		System.out.println("\t" + vector);
		}
		System.out.println("Singular values are: " + s);
		System.out.println("V factor is:\n" + V);
		jsc.stop();
	}
}
