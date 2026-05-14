package neurotik.encoding;

import tensor.Tensor;

import java.util.HashSet;
import java.util.List;

public abstract class Encoder {


    public abstract Tensor encode(char input);
    public abstract Tensor [] encode(String input);
    public abstract Tensor encode(char[] input);


    public abstract char decode(int input);

    public abstract HashSet<Tensor> parameters();

    public abstract List<Character> getVocab();

    public abstract int getEmbVecSize();
}
