package sparkML.ml;

import org.apache.spark.mllib.stat.test.BinarySample;
import org.apache.spark.mllib.stat.test.StreamingTest;
import org.apache.spark.mllib.stat.test.StreamingTestResult;
// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Seconds;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.util.Utils;

public class JavaStreamingTestExample {
	private static int timeoutCounter = 0;
	public static void main(String[] args) throws Exception {
	if (args.length != 3) {
	System.err.println("Usage: JavaStreamingTestExample " +
	"<dataDir> <batchDuration> <numBatchesTimeout>");
	System.exit(1);
	}
	String dataDir = "/home/bizruntime/new.txt";
	Duration batchDuration = Seconds.apply(5);
	int numBatchesTimeout = 100;
	SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("StreamingTestExample");
	JavaStreamingContext ssc = new JavaStreamingContext(conf, batchDuration);
	ssc.checkpoint(Utils.createTempDir(System.getProperty("java.io.tmpdir"), "spark").toString());
	// $example on$
	JavaDStream<BinarySample> data = ssc.textFileStream(dataDir).map(line -> {
	String[] ts = line.split(",");
	boolean label = Boolean.parseBoolean(ts[0]);
	double value = Double.parseDouble(ts[1]);
	return new BinarySample(label, value);
	});
	StreamingTest streamingTest = new StreamingTest()
	.setPeacePeriod(0)
	.setWindowSize(0)
	.setTestMethod("welch");
	JavaDStream<StreamingTestResult> out = streamingTest.registerStream(data);
	out.print();
	// $example off$
	// Stop processing if test becomes significant or we time out
	timeoutCounter = numBatchesTimeout;
	out.foreachRDD(rdd -> {
	timeoutCounter -= 1;
	boolean anySignificant = !rdd.filter(v -> v.pValue() < 0.05).isEmpty();
	if (timeoutCounter <= 0 || anySignificant) {
	rdd.context().stop();
	}
	});
	ssc.start();
	ssc.awaitTermination();
	}
	}

