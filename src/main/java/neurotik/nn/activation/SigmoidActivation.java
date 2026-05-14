package neurotik.nn.activation;

import tensor.Tensor;

public class SigmoidActivation implements Activation{
    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }
}
