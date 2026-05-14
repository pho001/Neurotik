package neurotik.nn.init;

import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;

public class NormalInit implements Initializer {

    double stdDev;
    double mean;
    public NormalInit(double stdDev,double mean){
        this.stdDev=stdDev;
        this.mean=mean;

    }
    @Override
    public Tensor init(int input,int output) {
        Random random = new Random();
        double[] data = new double[input * output];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian() * stdDev + mean;
        }
        return new Tensor(data, new int[]{input, output}, List.of(), "", DataType.FLOAT64).trainableParameter();
    }
}
