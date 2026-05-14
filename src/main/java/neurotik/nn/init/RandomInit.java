package neurotik.nn.init;

import neurotik.tensor.Tensor;

import java.util.HashSet;

public class RandomInit implements Initializer {

    public RandomInit(){

    }
    @Override
    public Tensor init(int input,int output) {
        Tensor out=new Tensor (input, output, new HashSet<>(),"").randTensor();
        return out;
    }
}
