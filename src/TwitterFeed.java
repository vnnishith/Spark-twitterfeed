import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import shapeless.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.stream.Collectors;

/**
 * A class which uses Spark to cluster given Twitter tweets by their geographic origins (coordinates), using
 * the K-means clustering algorithm
 */
class TwitterFeed {

    public static void  main (String[] args) {
        SparkConf conf = new SparkConf().setAppName("TwitterFeed").setMaster("local[4]");
        JavaSparkContext ctx = new JavaSparkContext(conf);
        ctx.setLogLevel("ERROR"); //optional : setting log level to ERROR mode
        JavaRDD<String> lines = ctx.textFile("twitter2D.txt", 1); // read from the given path all the lines
        JavaRDD<Vector> twitterFeedData = lines.map(
                (String s) -> {
                    String[] feedData = s.split(","); // split every line with the given regular expression to get the required coordinates
                    double[] coords= new double[2]; // declare a double to store the coordinates
                    for (int i = 0; i < 2; i++) // making it generic, so that if the model has to be trained with first n features
                        coords[i] = Double.parseDouble(feedData[i]); // converting string to double
                    return Vectors.dense(coords); // returns a vector
                }
        );
        // cache the training data, so that it can be retrieved quickly during computations
        twitterFeedData.cache();
        int numClusters = 4; //setting the number of clusters to be formed to 4
        int numIterations = 30; // setting the number of iterations
        // training the twitter feed data with the coordinates, with a  predefined number of clusters
        KMeansModel clusters = KMeans.train(twitterFeedData.rdd(), numClusters, numIterations);
        lines.mapToPair(l->{
            String[] feedData = l.split(","); // split the input data to get the coordinates and the actual tweet
            double[] latLong = new double[2];
            for (int i = 0; i < 2; i++) {
                latLong[i] = Double.parseDouble(feedData[i]);
            }
            //clusters.predict(Vectors.dense(latLong) predict as to which cluster the given coordinates belongs to
            // forming a Tuple2 so that the tweets can be sorted based on cluster index
            return new Tuple2<Integer, String>(clusters.predict(Vectors.dense(latLong)),feedData[feedData.length-1]);
        }).sortByKey() // sorting by cluster index
        .foreach(values -> {
            System.out.println("Tweet \""+values._2() + "\" is in cluster " + values._1()); // printing the tweet and the cluster index
        });
        ctx.stop(); //stopping the spark context
        ctx.close(); //closing the spark context


        JavaPairRDD<String, ArrayList<String>> tokenAnagramPairs = lines.mapToPair(s -> {
            String anagram = new StringBuffer(s).reverse().toString(); //reversing the string and considering it to be a anagram
            return new Tuple2(s, Arrays.asList(anagram));
        });
        JavaPairRDD<String, ArrayList<String>> grouped = tokenAnagramPairs.groupByKey().map(
                x -> {
                    HashSet<String> set = new HashSet<String>(x._2()); // consider unique values
                    ArrayList<String>  uniqueResult = new ArrayList<>(set); //convert back to list
                    return new Tuple2<>(x._1(),uniqueResult);
                }
            );


    }

}
