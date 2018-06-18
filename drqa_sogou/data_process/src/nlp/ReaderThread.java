package nlp;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import utils.BCConvert;

public class ReaderThread  implements Runnable{
	protected BlockingQueue<String> blockingQueue = null;
	public static String filePath;
	BufferedReader br = null;
	Boolean is_train = true;
	StanfordCoreNLP corenlp = new StanfordCoreNLP("StanfordCoreNLP-chinese.properties");

	  public ReaderThread(BlockingQueue<String> blockingQueue, Boolean is_train, BufferedReader br){
	    this.blockingQueue = blockingQueue;     
	    this.br = br;
	    this.is_train = is_train;
	  }
	  
	  
	
	 public Map<String, List<String>> wordFeature(String text)  {
		 Map<String, List<String>> features = new HashMap<>();
		 
		 Annotation document = new Annotation(text);  
    	 corenlp.annotate(document); 
    	 List<String> res_tokens = new ArrayList<String>();
    	 List<String> res_poss = new ArrayList<String>();
    	 List<String> res_ners = new ArrayList<String>();
    	 List<String> res_starts = new ArrayList<String>();
    	 List<String> res_ends = new ArrayList<String>();
    	 
    	 
   
         List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
         for(CoreMap sentence: sentences) {
         	List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
         	for(CoreLabel  token: tokens) {
         		String word = token.getString(TextAnnotation.class);
         		String pos = token.getString(PartOfSpeechAnnotation.class);
         		String ner = token.getString(NamedEntityTagAnnotation.class);
         		int start = token.beginPosition();
         		int end = token.endPosition();
         		res_tokens.add(word);
         		res_ners.add(ner);
         		res_poss.add(pos);
         		res_starts.add(start+"");
         		res_ends.add(end+"");
         		
         	}
         }
         features.put("ners", res_ners);
         features.put("poss", res_poss);
         features.put("tokens", res_tokens);
         features.put("starts", res_starts);
         features.put("ends", res_ends);
		 //System.out.println(features);
		 return features;
	 }

	  
	 
	@Override
	public void run(){

	     try {
	        
	            String buffer =null;
	            while((buffer=br.readLine())!=null){
	            	
	            	buffer=buffer.replaceAll("\r|\n", "");
	            	JSONObject jsonObj;
	            	
	            	jsonObj  = new JSONObject (buffer);
	            	String query = jsonObj.getString("query");
	            	query=BCConvert.qj2bj(query);
	            	
	            	String query_id = jsonObj.getString("query_id");
	            	
	            	Map<String, List<String>> question_feature = wordFeature(query);
//	            	Map result = new HashMap();
//	            	result.put("query_id", query_id);
//	            	result.put("question_tokens", question_feature.get("tokens"));
//	            	result.put("question_poss", question_feature.get("poss"));
//	            	result.put("question_ners", question_feature.get("ners"));
//	            	result.put("question_starts", question_feature.get("starts"));
//	            	result.put("question_ends", question_feature.get("ends"));
//	            	result.put("question", query);
	            	JSONObject obj = new JSONObject();
	            	obj.put("query_id", query_id);
	            	obj.put("question_tokens", question_feature.get("tokens"));
	            	obj.put("question_poss", question_feature.get("poss"));
	            	obj.put("question_ners", question_feature.get("ners"));
	            	obj.put("question_starts", question_feature.get("starts"));
	            	obj.put("question_ends", question_feature.get("ends"));
	            	obj.put("question", query);
	            	
	            	
	            	
	            	if (this.is_train) {
	            		String answer = jsonObj.getString("answer");
	            		obj.put("answer", answer);
	            	}
	            	
	            	JSONArray evidences = jsonObj.getJSONArray("passages");
	            	List <JSONObject>evidences_lists = new ArrayList<JSONObject>();
	            	
	            	for(int i=0; i< evidences.length();i++) {
	            		JSONObject evidence = (JSONObject) evidences.get(i);
	            		String e_key = evidence.get("passage_id") +"";
	            		String passage_text = (String) evidence.get("passage_text");
	            		passage_text=BCConvert.qj2bj(passage_text);
	            		
	            		Map<String, List<String>> evidence_feature;
						try {
							evidence_feature = wordFeature(passage_text);
						} catch (Exception e) {
							// TODO Auto-generated catch block
							System.out.println("badid__" + query_id);
							continue;
						}
						JSONObject evidence_map = new JSONObject();
//						System.out.println(evidence_feature.get("tokens"));
	            		evidence_map.put("e_key", query_id + "_" + e_key);
	            		evidence_map.put("tokens", evidence_feature.get("tokens"));
	            		evidence_map.put("poss", evidence_feature.get("poss"));
	            		evidence_map.put("ners", evidence_feature.get("ners"));
	            		evidence_map.put("starts", evidence_feature.get("starts"));
	            		evidence_map.put("ends", evidence_feature.get("ends"));
	            		evidence_map.put("text", passage_text);
//						Evidence e = new Evidence(query_id+"_"+e_key, passage_text,evidence_feature.get("tokens"),evidence_feature.get("poss"),
//								evidence_feature.get("ners"),evidence_feature.get("starts"),evidence_feature.get("ends"));
	            		
	            		evidences_lists.add(evidence_map);
	            		
	            	}
	            	obj.put("evidences", evidences_lists);
	   
	            	String out = obj.toString();
	            	
	            	
	                blockingQueue.put(out);
	                System.out.println("query_id : " +query_id);
	            }

	            blockingQueue.put("EOF");  //When end of file has been reached
	            System.out.println("test,,,");
	        } catch (FileNotFoundException e) {

	            e.printStackTrace();
	        } catch (IOException e) {

	            e.printStackTrace();
	        } catch(InterruptedException e){

	        }catch(JSONException e) {
	        	
	        }

		
	}

}
