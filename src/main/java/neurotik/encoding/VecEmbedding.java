package neurotik.encoding;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;


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
        Random random = new Random();
        double[] embeddingData = new double[lookup.size() * vecSize];
        for (int i = 0; i < embeddingData.length; i++) {
            embeddingData[i] = random.nextGaussian();
        }
        EMB = new Tensor(embeddingData, new int[]{lookup.size(), vecSize}, List.of(), "VecEmb", DataType.FLOAT64)
                .trainableParameter();

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
        Tensor indexTensor = new Tensor(new int[]{index}, new int[]{1}, List.of(), "embedding index", DataType.INT32);
        return EMB.gatherAxis(indexTensor, 0);

    }

    @Override
    public Tensor encode(char[] inputs){
        int [] indexes=new int[inputs.length];
        for (int i=0;i<inputs.length;i++){
            indexes[i]=this.lookup.indexOf(inputs[i]);
        }
        Tensor indexTensor = new Tensor(indexes, new int[]{indexes.length}, List.of(), "embedding indexes", DataType.INT32);
        return EMB.gatherAxis(indexTensor, 0);
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
