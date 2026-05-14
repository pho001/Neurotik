package neurotik.nn.init;

import neurotik.tensor.Tensor;

import java.util.HashSet;

public class UniformInit implements Initializer {

    double upper;
    double lower;
    public UniformInit(double upper, double lower){
        this.upper=upper;
        this.lower=lower;

    }
    @Override
    public Tensor init(int input,int output) {
        Tensor out=new Tensor(input,output,new HashSet<>(),"").randTensor();
        out= out.muleach(upper-lower).addEach(lower);
        out.noGradientPassdown();
        return out;
    }
}
