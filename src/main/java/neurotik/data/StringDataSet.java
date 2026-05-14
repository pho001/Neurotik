package neurotik.data;

import neurotik.encoding.Encoder;
import tensor.Tensor;

import java.util.*;

public class StringDataSet extends DataSet<String> {

    int seqLength;

    public StringDataSet(List<String> lines){
        super(lines);
    }
    public StringDataSet sortDesc(){
        this.data.sort((s1, s2) -> Integer.compare(s2.length(), s1.length()));
        return this;
    }

    @Override
    public StringDataSet setSequences(int context,boolean usePadding){
        this.seqLength=context;
        List<String> ngrams = new ArrayList<>();
        List<StringDataRecord> record=new ArrayList<>();

        for (String line : this.data) {
            String[] words = line.split(" ");

            for (String word : words) {
                //word = word.toLowerCase();

                if (usePadding) {
                    word=".".repeat(context-1)+word+".";
                    for (int i = 0; i < word.length()-(context-1); i++) {

                        ngrams.add(word.substring(i, i + context));
                    }
                }
                else{
                    //word = word+".";
                    word=".".repeat(context-1)+word+".";
                    for (int i = 1; i < word.length() - (context - 1); i++) {
                        if (i <= context - 1)
                            ngrams.add(word.substring(context - 1, i + context));
                        else
                            ngrams.add(word.substring(i, i + context));
                    }

                }
            }

        }

        return new StringDataSet(ngrams);
    }





    public String uniqueCharacters(){
        Set<Character> unique = new TreeSet<>();
        for (String word:this.data){
            for (char c:word.toCharArray()) {
                unique.add(c);
            }
        }
        String output="";
        for (char character:unique){
            output+=character;
        }
        return output;

    }

    @Override
    public StringDataSet getSubSeq(int startOffset, int endOffset){
        List<String> out= new ArrayList<>();
        int relIndex;
        for (String s : this.data) {
            int startIndex=startOffset;
            int endIndex=s.length()+endOffset;
            if (startIndex < 0 || startIndex > s.length() || endIndex < 0 || endIndex > s.length() || startIndex > endIndex) {
                throw new IllegalArgumentException("Invalid offsets provided.");
            }
            out.add(s.substring(startIndex,endIndex));
        }
        return new StringDataSet(out);
    }

    @Override
    public StringDataSet getSubSeq(int endOffset){
        List<String> out= new ArrayList<>();
        int relIndex;
        for (String s : this.data) {
            int startIndex=s.length()+endOffset;
            int endIndex=s.length();
            if (startIndex < 0 || startIndex > s.length() || endIndex < 0 || endIndex > s.length() || startIndex > endIndex) {
                throw new IllegalArgumentException("Invalid offsets provided.");
            }
            out.add(s.substring(startIndex,endIndex));
        }
        return new StringDataSet(out);
    }

    @Override
    public StringDataSet padSortedSequences(String paddingString){
        List <String> sortedList=this.data.stream()
                .sorted(Comparator.comparingInt(String::length).reversed())
                .toList();
        int first=sortedList.get(0).length();
        int last=sortedList.get(sortedList.size()-1).length();
        List <String> output=new ArrayList<>();
        if (first==last){
            output=sortedList; //no padding needed
        }
        else {
            for (String word : sortedList) {
                String tokens = paddingString.repeat(first - word.length());
                output.add(word + tokens);
            }
        }
        return new StringDataSet(output);
    }

    @Override
    public int[] getMask() {
        int[] mask=new int[this.data.size()];
        for (int i=0;i<this.data.size();i++){
            mask[i]=this.data.get(i).length();
        }
        return mask;
    }

    public Tensor[] encode(Encoder enc){

        int length=this.data.get(0).length();
        Tensor[] out=new Tensor[length];
        for (int i=0;i<length;i++){
            Tensor contextLayer=null;
            Tensor encoded=null;
            char[] chars = new char [this.data.size()];
            int j=0;
            for (String s : this.data) {
                chars[j]=s.charAt(i);
                j++;
            }
            contextLayer=enc.encode(chars);
            out[i]=contextLayer;
        }
        return out;
    }

    public Tensor encode(Encoder enc,int context){
        Tensor out=null;
        char[] chars = new char [this.data.size()];
        int j=0;
        for (String s : this.data) {
            chars[j]=s.charAt(context);
            j++;
        }
        out=enc.encode(chars);
        return out;
    }

    @Override
    public DataSet<String> createInstance(List<String> data) {
        return new StringDataSet(data);
    }




}
