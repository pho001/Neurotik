package neurotik.nn.init;

import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;

public class XavierInit implements Initializer {

    public XavierInit(){

    }

    @Override
    public Tensor init(int input,int output) {
        double stdDev = Math.sqrt(2.0 / (input + output));
        Random random = new Random();
        double[] data = new double[input * output];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian() * stdDev;
        }
        return new Tensor(data, new int[]{input, output}, List.of(), "", DataType.FLOAT64).trainableParameter();
    }
}
