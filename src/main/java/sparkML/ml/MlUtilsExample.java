package sparkML.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class MlUtilsExample {
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[2]");
	    JavaSparkContext jsc = new JavaSparkContext(sparkConf);
	    // $example on$
	    String path = "/home/bizruntime/test.txt";
	    JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
	    System.out.println(inputData.collect());
	}
}
