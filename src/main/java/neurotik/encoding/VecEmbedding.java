package neurotik.encoding;

import neurotik.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;


public class VecEmbedding extends Encoder{

    Tensor EMB;
    List<Character> lookup;
    int vecSize;
    int encodedTokenSize;



    public VecEmbedding(String charSet, int vecSize){

        this.lookup=new ArrayList<>();
        for (int i=0;i<charSet.length();i++){
            lookup.add(charSet.charAt(i));

        }
        EMB=new Tensor(lookup.size(),vecSize, new HashSet<>(),"VecEmb").randTensor();

        this.vecSize=vecSize;
        this.encodedTokenSize=vecSize;
    }

    @Override
    public char decode(int input){
        char output;
        output= lookup.get(input);
        return output;
    }


    @Override
    public Tensor encode(char input){
        int index=this.lookup.indexOf(input);
        return EMB.mapVec(index);

    }

    @Override
    public Tensor encode(char[] inputs){
        int [] indexes=new int[inputs.length];
        for (int i=0;i<inputs.length;i++){
            indexes[i]=this.lookup.indexOf(inputs[i]);
        }
        return EMB.mapVec(indexes);
    }

    @Override
    public Tensor[] encode(String input){
        Tensor[] enc=new Tensor [input.length()];
        for (int i=0;i<input.length();i++){
            char c=(char) input.charAt(i);
            enc[i]=encode((char) c);
        }

        return enc;
    }


    @Override
    public List<Character> getVocab() {
        return lookup;
    }

    @Override
    public int getEmbVecSize() {
        return this.encodedTokenSize;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(EMB);
        return params;
    }





}
