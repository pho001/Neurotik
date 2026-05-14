package neurotik.nn.layers;

import neurotik.nn.activation.Activation;
import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.tensor.Tensor;

import java.util.HashSet;

public class ActivationLayer extends Layer
{

    Activation activation;
    Tensor [] out=null;

    public ActivationLayer(Activation activation){
        this.activation=activation;
    }


    @Override
    public Tensor[] forward(Tensor[] input) {
        this.out=new Tensor [input.length];
        for (int i=0;i<input.length;i++)
        {
            this.out[i]=activation.forward(input[i]);
        }
        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        return new HashSet<>();
    }

    @Override
    public void initParameters(Initializer init) {

    }


}
