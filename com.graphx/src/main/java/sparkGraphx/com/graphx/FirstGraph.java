package sparkGraphx.com.graphx;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;
import scala.reflect.ClassTag$;
import scala.runtime.AbstractFunction1;

public class FirstGraph implements Serializable{

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("graphExample").setMaster("local[*]");
		JavaSparkContext ctx = new  JavaSparkContext(conf);
		
		List<Tuple2<Object, User>> list = new ArrayList<Tuple2<Object, User>>();
		list.add(new Tuple2<Object, User>(3l, new User("rxin", "student")));
		list.add(new Tuple2<Object, User>(7l, new User("jgonzal", "postdoc")));
		list.add(new Tuple2<Object, User>(5l, new User("franklin", "prof")));
		list.add(new Tuple2<Object, User>(2l, new User("istoica", "prof")));
		list.add(new Tuple2<Object, User>(4l, new User("peter", "student")));
		
		JavaRDD<Tuple2<Object, User>> vertexRDD =	ctx.parallelize(list);
		
		List<Edge<String>> edgeList = new ArrayList<Edge<String>>();
		edgeList.add(new Edge<String>(3l, 7l, "collab"));
		edgeList.add(new Edge<String>(5l, 3l, "advisor"));
		edgeList.add(new Edge<String>(2l, 5l, "colleague"));
		edgeList.add(new Edge<String>(5l, 7l, "pi"));
		edgeList.add(new Edge<String>(4L, 0L, "student"));	
		edgeList.add(new Edge<String>(5L, 0L, "colleague"));
		
		JavaRDD<Edge<String>> edgeRDD =	ctx.parallelize(edgeList);
		
		User defaultUser = new User("defaultUser", "unknown");
		Graph<User,String> graph = Graph.<User,String>apply(vertexRDD.rdd(), edgeRDD.rdd(), defaultUser, StorageLevel.MEMORY_AND_DISK(),
				StorageLevel.MEMORY_AND_DISK(), ClassTag$.MODULE$.<User>apply(User.class), ClassTag$.MODULE$.<String>apply(String.class));

		System.out.println(graph.ops().numEdges());
		
		System.out.println(graph.ops().numVertices());
		
		System.out.println(graph.vertices().filter(new Filter()).count());
	
		}
	
	
}
class Filter extends  AbstractFunction1<Tuple2<Object,User>,Object> implements Serializable{
	public Object apply(Tuple2<Object, User> arg0) {
		return arg0._2.getName().equals("rxin");
	}
}