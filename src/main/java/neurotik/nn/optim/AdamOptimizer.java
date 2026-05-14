package neurotik.nn.optim;

import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.IdentityHashMap;
import java.util.Map;

public class AdamOptimizer extends Optimizer{
    // Parametry algoritmu Adam
    private double beta1;
    private double beta2;
    private double learningRate;
    private double epsilon;
    private final Map<Tensor, double[]> means = new IdentityHashMap<>();
    private final Map<Tensor, double[]> variances = new IdentityHashMap<>();


    // Časový krok

    public AdamOptimizer(double beta1, double beta2, double learningRate, double epsilon){
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    @Override
    public void update(Tensor parameter,int timeStep) {
        Tensor gradient = parameter.getGradient();
        if (gradient == null) {
            return;
        }

        double[] data = parameter.toDoubleArrayCopy();
        double[] grad = gradient.toDoubleArrayCopy();
        double[] mean = means.computeIfAbsent(parameter, ignored -> new double[data.length]);
        double[] variance = variances.computeIfAbsent(parameter, ignored -> new double[data.length]);

        for (int i = 0; i < data.length; i++) {
            mean[i] = beta1 * mean[i] + (1 - beta1) * grad[i];
            variance[i] = beta2 * variance[i] + (1 - beta2) * Math.pow(grad[i], 2);

            double mHat = mean[i] / (1 - Math.pow(beta1, timeStep));
            double vHat = variance[i] / (1 - Math.pow(beta2, timeStep));

            data[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
        parameter.setData(data);
        TensorInternalAccess.clearGradient(parameter);
    }
}
