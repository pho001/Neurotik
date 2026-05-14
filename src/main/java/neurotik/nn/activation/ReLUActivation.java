package neurotik.nn.activation;

import tensor.Tensor;

public class ReLUActivation implements  Activation{
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
