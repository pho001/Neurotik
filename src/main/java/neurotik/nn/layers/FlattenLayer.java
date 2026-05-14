package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import tensor.Tensor;

import java.util.HashSet;

public class FlattenLayer extends Layer{

    int factor;
    Tensor out= null;

    public FlattenLayer(int factor){
        this.factor=factor;
    }




    public Tensor forward(Tensor input){
        int[] shape = input.getShape();
        if (shape.length == 2) {
            this.out = input.transpose();
            return this.out;
        }
        if (shape.length != 3) {
            throw new IllegalArgumentException("FlattenLayer expects rank 2 or 3 input.");
        }
        int time = shape[0];
        int batch = shape[1];
        int features = shape[2];
        this.out = input.permute(1, 0, 2).reshape(batch, time * features);
        return this.out;
    }



    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }

    @Override
    public void initParameters(Initializer init) {

    }
}
