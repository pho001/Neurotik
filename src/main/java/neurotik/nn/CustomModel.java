package neurotik.nn;

import neurotik.data.DataLoader;
import neurotik.encoding.Encoder;
import neurotik.nn.init.Initializer;
import neurotik.nn.optim.Optimizer;
import tensor.Tensor;

import java.util.HashSet;

public class CustomModel extends Model{
    Layers layers;
    int contextLength;

    //last layer must be output layer
    int [] hiddenLayers;
    boolean useBias;

    Encoder encoder;

    public CustomModel(int contextLength, int featuresIn, int [] hiddenLayers, int featuresOut, Encoder encoder, boolean useBias, Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,encoder,useBias,init);
    }

    public CustomModel(int contextLength, int featuresIn, int [] hiddenLayers, int featuresOut, boolean useBias, Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,useBias,init);
    }
    @Override
    public Layers topology(){
        return layers;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet<Tensor> params=new HashSet<>();
        for(Layer layer:layers.layers){
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public<T> void train(DataLoader <T> ds, int batchSize, int epochs, Optimizer optimizer) {

    }



    @Override
    public void generate(int samples) {

    }

}
