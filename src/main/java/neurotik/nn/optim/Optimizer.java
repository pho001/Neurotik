package neurotik.nn.optim;

import tensor.Tensor;

public abstract class Optimizer {
    public Optimizer(){

    }
    public abstract void update(Tensor Parameter,int epoch);
}
