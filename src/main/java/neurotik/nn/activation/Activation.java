package neurotik.nn.activation;

import neurotik.tensor.Tensor;

public interface Activation {
    public abstract Tensor forward(Tensor input);

}
