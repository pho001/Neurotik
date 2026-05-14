package neurotik.encoding;

import neurotik.tensor.Tensor;

import java.util.HashSet;
import java.util.List;

public interface IEncoder {

    public Tensor encode(char input);
    public Tensor encode(String input);
    public Tensor encode(char[] input);

    public char decode(int input);

    public HashSet<Tensor> parameters();

    public List<Character> getVocab();

}
