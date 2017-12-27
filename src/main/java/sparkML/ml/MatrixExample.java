package sparkML.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.QRDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class MatrixExample {
	public static void main(String[] args) {
		//matrices();
		//indexMatrices();
		coordinateMatrixEx();
	}
	public static void sparkCon() {
		SparkConf conf = new SparkConf().setAppName("MLib Examples").setMaster("local[2]");
	    //SparkContext sc = new SparkContext(conf);
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	}
	public static void matrices()
	{
		SparkConf conf = new SparkConf().setAppName("MLib Examples").setMaster("local[2]");
	    //SparkContext sc = new SparkContext(conf);
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	    
	    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
	    Matrix dm = Matrices.dense(3, 2, new double[] {1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

	    // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
	    Matrix sm = Matrices.sparse(3, 2, new int[] {0, 1, 3}, new int[] {0, 2, 1}, new double[] {9, 6, 8});
	    System.out.println(dm + " " + sm);
	    
	    //RowMatrix
	    List<Vector> data = Arrays.asList(
	            Vectors.dense(150,60, 35), Vectors.dense(300, 200, 600));
	    
	    
	    JavaRDD<Vector> rows = jsc.parallelize(data);
	    RowMatrix mat = new RowMatrix(rows.rdd());
	 // Get its size.
	    long m = mat.numRows();
	    long n = mat.numCols();
	    System.out.println(m + " " +n);

	 // QR decomposition 
	   // QRDecomposition<RowMatrix, Matrix> result = mat.tallSkinnyQR(true);
	}
	public static void indexMatrices() 
	{
		SparkConf conf = new SparkConf().setAppName("MLib Examples").setMaster("local[2]");
	    //SparkContext sc = new SparkContext(conf);
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	    IndexedRow iRow1 = new IndexedRow(0, Vectors.sparse(2, new int[]{0}, new double[]{1.0}));
        IndexedRow iRow2 = new IndexedRow(1, Vectors.sparse(2, new int[]{0, 1}, new double[]{3.0, 2.0}));
        IndexedRow iRow3 = new IndexedRow(2, Vectors.sparse(2, new int[]{1}, new double[]{6.0}));
	    
	    
        List<IndexedRow> inputList = new ArrayList<IndexedRow>();
        inputList.add(iRow1);
        inputList.add(iRow2);
        inputList.add(iRow3);
        
        //JavaSparkContext context = new JavaSparkContext();
        JavaRDD<IndexedRow> rows = jsc.parallelize(inputList);

        IndexedRowMatrix matrix = new IndexedRowMatrix(rows.rdd());
        System.out.println("Row count: " + matrix.numRows() + ", Column count: " + matrix.numCols());
		
	}
	 public static void coordinateMatrixEx() {
		 	SparkConf conf = new SparkConf().setAppName("MLib Examples").setMaster("local[2]");
		    //SparkContext sc = new SparkContext(conf);
		    JavaSparkContext jsc = new JavaSparkContext(conf);
	        MatrixEntry m1 = new MatrixEntry(4, 8, 1.0);
	        MatrixEntry m3 = new MatrixEntry(9, 6, 3.0);
	        MatrixEntry m4 = new MatrixEntry(8, 16, 2.0);
	        MatrixEntry m6 = new MatrixEntry(88, 11, 6.0);

	        List<MatrixEntry> matrixEntries = Arrays.asList(new MatrixEntry[]{m1, m3, m4, m6});
	        //JavaSparkContext javaSparkContext = new JavaSparkContext();
	        JavaRDD<MatrixEntry> matrixEntryRDD = jsc.parallelize(matrixEntries);

	        CoordinateMatrix coordinateMatrix = new CoordinateMatrix(matrixEntryRDD.rdd());
	        System.out.println("Row count: " + coordinateMatrix.numRows());
	        System.out.println("Column count: " + coordinateMatrix.numCols());
	    }
	 /*public static void blockMat() {
		 	MatrixEntry m1 = new MatrixEntry(4, 8, 1.0);
	        MatrixEntry m3 = new MatrixEntry(9, 6, 3.0);
	        MatrixEntry m4 = new MatrixEntry(8, 16, 2.0);
	        MatrixEntry m6 = new MatrixEntry(88, 11, 6.0);
	        
	        List<MatrixEntry> matrixEntries = Arrays.asList(new MatrixEntry[]{m1, m3, m4, m6});
	        JavaSparkContext jsc = new JavaSparkContext();
	        
	        JavaRDD<MatrixEntry> matrixEntryRDD = jsc.parallelize(matrixEntries);
	        CoordinateMatrix coordMat = new CoordinateMatrix( matrixEntries.matrixEntryRDD());
	        BlockMatrix matA = coordMat.toBlockMatrix().cache();
	     // Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
	     // Nothing happens if it is valid.
	        matA.validate();
	        BlockMatrix ata = matA.transpose().multiply(matA);
	 }*/
}
