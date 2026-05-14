package neurotik.encoding;

import java.util.List;

public class EncoderFactory
{
    public static Encoder createInteger(List<Character> lookup) {
        return new IntegerEncoder(lookup);
    }

    public static Encoder createOnehot(List<Character> lookup){
        return new OneHotEncoder(lookup);
    }

    public static Encoder createInteger(String lookup){
        return new IntegerEncoder(lookup);
    }

    public static Encoder createOnehot(String lookup){
        return new OneHotEncoder(lookup);
    }

    public static Encoder createVecEmb(String lookup, int vecSize){
        return new VecEmbedding(lookup,vecSize);
    }

    public static Encoder createWord2Vec(String lookup, int vecSize){
        //@TODO: to be implemented
        return null;
    }


}
