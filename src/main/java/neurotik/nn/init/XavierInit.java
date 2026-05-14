package neurotik.nn.init;

import neurotik.tensor.Tensor;

import java.util.HashSet;

public class XavierInit implements Initializer {

    public XavierInit(){

    }

    @Override
    public Tensor init(int input,int output) {
        double stdDev = Math.sqrt(2.0 / (input + output));
        Tensor out=new Tensor(input,output,new HashSet<>(),"").randTensor().muleach(stdDev);
        out.noGradientPassdown();
        return out;
    }
}
