package neurotik.nn.activation;

import tensor.Tensor;

public interface Activation {
    public abstract Tensor forward(Tensor input);

}
