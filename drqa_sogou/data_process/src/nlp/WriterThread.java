package nlp;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.concurrent.BlockingQueue;

public class WriterThread implements Runnable{

  protected BlockingQueue<String> blockingQueue = null;
  private int count = 0;
  private int threadsize = 0;
  private String outfile;
  public WriterThread(BlockingQueue<String> blockingQueue, int threadsize, String outfile){
    this.blockingQueue = blockingQueue;    
    this.threadsize = threadsize;
    this.outfile = outfile;
  }

  @Override
  public void run() {
    PrintWriter writer = null;

    try {
        writer = new PrintWriter(new OutputStreamWriter(  new FileOutputStream(outfile),"UTF-8"));

        while(true){
            String buffer = blockingQueue.take();
            //System.out.println(buffer);
            //Check whether end of file has been reached
            if(buffer.equals("EOF")){ 
            	// System.out.println("写出EOF");
               count += 1;
               if(count == this.threadsize) {
               	break;
               }
               continue;
            }
            
            writer.println(buffer);
            System.out.println(buffer);
        }               


    } catch (FileNotFoundException e) {

        e.printStackTrace();
    } catch(InterruptedException e){
    	e.printStackTrace();
    } catch (UnsupportedEncodingException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}finally{
        writer.close();
    } 

  }

}
