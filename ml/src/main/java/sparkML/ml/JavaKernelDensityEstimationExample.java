package sparkML.ml;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.stat.KernelDensity;

public class JavaKernelDensityEstimationExample {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaKernelDensityEstimationExample").setMaster("local[2]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		// $example on$
		// an RDD of sample data
		JavaRDD<Double> data = jsc.parallelize(
		Arrays.asList(1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0));
		// Construct the density estimator with the sample data
		// and a standard deviation for the Gaussian kernels
		KernelDensity kd = new KernelDensity().setSample(data).setBandwidth(3.0);
		// Find density estimates for the given values
		double[] densities = kd.estimate(new double[]{-1.0, 2.0, 5.0});
		System.out.println(Arrays.toString(densities));
		// $example off$
		jsc.stop();
		}
		
}
// O/P :- [0.04145944023341912, 0.07902016933085627, 0.08962920127312338]