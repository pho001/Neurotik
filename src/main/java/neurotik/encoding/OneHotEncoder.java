package neurotik.encoding;

import neurotik.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


public class OneHotEncoder extends Encoder{
    List<Character> lookup;
    int encodedTokenSize;

    public OneHotEncoder(List<Character> charset){
        this.lookup=charset;
        encodedTokenSize= charset.size();
    }

    public OneHotEncoder(String charSet){
        this.lookup=new ArrayList<>();
        for (int i=0;i<charSet.length();i++){
            lookup.add(charSet.charAt(i));
        }
    }


    @Override
    public Tensor encode(char input) {
        double [][] oneHot=new double [1][lookup.size()];
        oneHot[0][this.lookup.indexOf(input)]=1;
        Tensor out=new Tensor(oneHot,new HashSet<>(),"oneHot");
        return out;
    }

    @Override
    public char decode(int input) {
        char output;
        output= lookup.get(input);
        return output;
    }
    @Override
    public Tensor [] encode(String input){
        Tensor[] enc=new Tensor [input.length()];
        for (int i=0;i<input.length();i++){
            char c=(char) input.charAt(i);
            enc[i]=encode((char) c);
        }

        return enc;
    }

    @Override
    public Tensor encode(char[] inputs){

        double [][] enc=new double[inputs.length][lookup.size()];
        for (int i=0;i<inputs.length;i++){
           enc[i][this.lookup.indexOf(inputs[i])]=1;
        }
        Tensor out=new Tensor(enc,new HashSet<>(),"");
        return out;
    }

    @Override
    public List<Character> getVocab() {
        return lookup;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }

    @Override
    public int getEmbVecSize() {
        return lookup.size();
    }
}
