package neurotik.nn.activation;

import neurotik.tensor.Tensor;

public class TanhActivation implements Activation {
    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }
}


