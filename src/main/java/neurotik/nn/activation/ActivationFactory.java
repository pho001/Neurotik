package neurotik.nn.activation;

public class ActivationFactory {
    public static Activation tanh(){
        return new TanhActivation();
    }
    public static Activation ReLU(){
        return new ReLUActivation();
    }

    public static Activation sigmoid(){
        return new SigmoidActivation();
    }
}
