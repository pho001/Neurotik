package neurotik.encoding;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class IntegerEncoder extends Encoder{

    List<Character>   lookup;
    int encodedTokenSize;
    public IntegerEncoder(List<Character> lookup){

        this.lookup=lookup;
    }


    public IntegerEncoder(String charSet){
        this.lookup=new ArrayList<>();
        for (int i=0;i<charSet.length();i++){
            lookup.add(charSet.charAt(i));
        }
    }





    @Override
    public Tensor encode(char input) {
        double[][] intEnc=new double [1][1];
        intEnc[0][0]=lookup.indexOf(input);
        Tensor out= new Tensor(intEnc, List.of(), "intEnc", DataType.FLOAT64);
        return out;
    }

    @Override
    public Tensor [] encode(String input){
        Tensor[] enc=new Tensor [input.length()];
        for (int i=0;i<input.length();i++){
            char c=(char) input.charAt(i);
            enc[i]=encode((char) c);

            /*
            if (enc==null)
                enc=encode((char)c);
            else
                enc=enc.join(encode(c), Tensor.Join.RIGHT);

             */
        }

        return enc;
    }

    @Override
    public Tensor encode(char[] inputs){
        Tensor enc=null;
        for (int i=0;i<inputs.length;i++){
            char c=(char) inputs[i];
            if (enc==null)
                enc=encode((char)c);
            else
                enc=Tensor.concat(1, enc, encode(c));
        }

        return enc;
    }

    @Override
    public char decode(int input) {
        char output;
        output=lookup.get(input);
        return output;
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
        return this.encodedTokenSize;
    }

}
