package neurotik.nn.init;

import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;

public class KaimingInit implements Initializer {


    public KaimingInit(){

    }

    @Override
    public Tensor init(int input,int output) {
       double stDev=Math.sqrt(2.0 / input);
       Random random = new Random();
       double[] data = new double[input * output];
       for (int i = 0; i < data.length; i++) {
           data[i] = random.nextGaussian() * stDev;
       }
       return new Tensor(data, new int[]{input, output}, List.of(), "", DataType.FLOAT64).trainableParameter();
    }
}
