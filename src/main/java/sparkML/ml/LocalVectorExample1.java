package sparkML.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class LocalVectorExample1 {
	public static void main(String[] args) {
		String inputFileName = "/home/bizruntime/test.txt";
		SparkConf conf = new SparkConf().setAppName("MLib Examples").setMaster("local[2]");
	    SparkContext sc = new SparkContext(conf);
	    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);
		
		// Create a dense vector (1.0, 0.0, 3.0).
		Vector dv = Vectors.dense(1.0, 0.0, 3.0);
		// Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
		Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
		System.out.println(dv + "    " + sv);
		LabeledPoint pos = new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0));

		// Create a labeled point with a negative label and a sparse feature vector.
		LabeledPoint neg = new LabeledPoint(0.0, Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0}));
		System.out.println(pos + "  " + neg);
		//JavaRDD<LabeledPoint> file = sc.textFile(inputFileName);
		
		JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), inputFileName).toJavaRDD();
		System.out.println(inputData);
		
	}
}
