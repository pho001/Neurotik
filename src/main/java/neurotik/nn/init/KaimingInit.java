package neurotik.nn.init;

import neurotik.tensor.Tensor;

import java.util.HashSet;

public class KaimingInit implements Initializer {


    public KaimingInit(){

    }

    @Override
    public Tensor init(int input,int output) {
       double stDev=Math.sqrt(2.0 / input);
       Tensor out=new Tensor(input,output,new HashSet<>(),"").randTensor().muleach(stDev);
       out.noGradientPassdown();
       return out;
    }
}
