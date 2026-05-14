package neurotik.nn.optim;

import tensor.Tensor;
import tensor.TensorInternalAccess;

public class GDOptimizer extends Optimizer{
    double learningRate;
    public GDOptimizer(double learningRate){

        this.learningRate=learningRate;
    }

    @Override
    public void update(Tensor Parameter,int epoch) {
        Tensor gradient = Parameter.getGradient();
        if (gradient == null) {
            return;
        }
        double[] data = Parameter.toDoubleArrayCopy();
        double[] grad = gradient.toDoubleArrayCopy();
        for (int i = 0; i < data.length; i++) {
            data[i] += -learningRate * grad[i];
        }
        Parameter.setData(data);
        TensorInternalAccess.clearGradient(Parameter);
    }
}
