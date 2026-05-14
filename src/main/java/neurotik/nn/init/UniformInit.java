package neurotik.nn.init;

import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;

public class UniformInit implements Initializer {

    double upper;
    double lower;
    public UniformInit(double upper, double lower){
        this.upper=upper;
        this.lower=lower;

    }
    @Override
    public Tensor init(int input,int output) {
        Random random = new Random();
        double[] data = new double[input * output];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian() * (upper - lower) + lower;
        }
        return new Tensor(data, new int[]{input, output}, List.of(), "", DataType.FLOAT64).trainableParameter();
    }
}
