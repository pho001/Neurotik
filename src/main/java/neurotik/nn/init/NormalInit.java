package neurotik.nn.init;

import neurotik.tensor.Tensor;

import java.util.HashSet;

public class NormalInit implements Initializer {

    double stdDev;
    double mean;
    public NormalInit(double stdDev,double mean){
        this.stdDev=stdDev;
        this.mean=mean;

    }
    @Override
    public Tensor init(int input,int output) {
        Tensor out=new Tensor(input,output,new HashSet<>(),"").randTensor();
        out=out.muleach(stdDev).addEach(mean);
        out.noGradientPassdown();
        return out;
    }
}
