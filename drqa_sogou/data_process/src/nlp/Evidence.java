package nlp;

import java.util.List;

public class Evidence {
	private String e_key;
	private List<String> tokens;
	private List<String> starts;
	private List<String> ends;
	private List<String> poss;
	private List<String> ners;
	private String text;
	public Evidence(String e_key,String text,List<String> tokens,List<String> poss,List<String> ners, List<String> starts, List<String> ends) {
		this.e_key = e_key;
		this.text = text;
		this.tokens = tokens;
		this.ners =ners;
		this.poss = poss;
		this.starts =starts;
		this.ends = ends;
	}
	public String getE_key() {
		return e_key;
	}
	public void setE_key(String e_key) {
		this.e_key = e_key;
	}
	public List<String> getTokens() {
		return tokens;
	}
	public void setTokens(List<String> tokens) {
		this.tokens = tokens;
	}
	public List<String> getStarts() {
		return starts;
	}
	public void setStarts(List<String> starts) {
		this.starts = starts;
	}
	public List<String> getEnds() {
		return ends;
	}
	public void setEnds(List<String> ends) {
		this.ends = ends;
	}
	public List<String> getPoss() {
		return poss;
	}
	public void setPoss(List<String> poss) {
		this.poss = poss;
	}
	public List<String> getNers() {
		return ners;
	}
	public void setNers(List<String> ners) {
		this.ners = ners;
	}
	public String getText() {
		return text;
	}
	public void setText(String text) {
		this.text = text;
	}
	
}
