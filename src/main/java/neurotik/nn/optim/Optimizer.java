package neurotik.nn.optim;

import neurotik.tensor.Tensor;

public abstract class Optimizer {
    public Optimizer(){

    }
    public abstract void update(Tensor Parameter,int epoch);
}
