package nlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.text.ParseException;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class NLPTool {
	public static void main(String[] args) throws ParseException, FileNotFoundException {
		
//		if(args.length < 3) {
//			System.out.println("Usage: java NLPTool inputfile istrain(0 for test,1 for train) outputfile");
//			return;
//		}
//		String filepath = args[0];
//		int is_train = Integer.parseInt(args[1]);
		Boolean istrain=true;
//		if (is_train==0) {
//			istrain=false;
//		}
//		String outfile = args[2];
		
		BlockingQueue<String> queue = new ArrayBlockingQueue<String>(1024);
		String filepath ="/media/iscas/linux/fym/data/raw_data/train_factoid_2.json";
		
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
	    ReaderThread reader = new ReaderThread(queue,istrain, br);
	    String outfile = "/media/iscas/linux/fym/data/java_pre_data/train_factoid_2_java.json";
	    int threadsize = 8;
	    WriterThread writer = new WriterThread(queue,threadsize,outfile);
	    for(int i=0;i<threadsize;i++) {
	    	new Thread(reader).start();
	    }
	    new Thread(writer).start();

	}
}
