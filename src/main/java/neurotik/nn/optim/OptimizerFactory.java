package neurotik.nn.optim;

public class OptimizerFactory {
    public static Optimizer AdamOptimizer(double learningRate){
        return new AdamOptimizer(0.9, 0.999, learningRate, 1e-8);
    }
    public static Optimizer GDOptimizer(double learningRate){
        return new GDOptimizer(learningRate);
    }
}
