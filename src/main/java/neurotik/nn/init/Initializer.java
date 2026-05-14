package neurotik.nn.init;

import neurotik.tensor.Tensor;

public interface Initializer {

    public abstract Tensor init(int input,int output);
}
