package neurotik.nn;

import neurotik.nn.activation.Activation;
import neurotik.nn.layers.ActivationLayer;
import neurotik.nn.layers.BatchNormLayer;
import neurotik.nn.layers.FlattenLayer;
import neurotik.nn.layers.GRULayer;
import neurotik.nn.init.Initializer;
import neurotik.nn.layers.LinearLayer;
import neurotik.nn.layers.LstmLayer;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class Layers {
    public List<Layer> layers;
    public Layers (){
        layers=new ArrayList<>();
    }

    public Layers add(Layer layer){
        layers.add(layer);
        return this;
    }

    public Layers addLinear(int features_in, int features_out, boolean useBias,Initializer init){
        layers.add(LayerFactory.LinearLayer(features_in,features_out,useBias,init));
        return this;
    }

    public Layers addActivation(Activation activation){
        layers.add(LayerFactory.ActivationLayer(activation));
        return this;
    }

    public Layers addBatchNorm(int features){
        layers.add(LayerFactory.BatchNormLayer(features));
        return this;
    }

    public Layers addBatchNorm(int features, int channelAxis){
        layers.add(LayerFactory.BatchNormLayer(features, channelAxis));
        return this;
    }

    public Layers addBatchNorm(int features, int channelAxis, double epsilon){
        layers.add(LayerFactory.BatchNormLayer(features, channelAxis, epsilon));
        return this;
    }

    public Layers addBatchNorm(int features, int channelAxis, double epsilon, double momentum){
        layers.add(LayerFactory.BatchNormLayer(features, channelAxis, epsilon, momentum));
        return this;
    }

    public Layers addFlatten(int factor){
        layers.add(LayerFactory.FlattenLayer(factor));
        return this;
    }

    public Layers addOneDirectionalRNN(int inputSize, int hiddenSize, boolean useBias,Initializer init){
        layers.add(LayerFactory.OneDirectionalRNN(inputSize,hiddenSize,useBias,init));
        return this;
    }

    public Layers addLstmLayer(int inputSize, int hiddenSize, boolean useBias,Initializer init){
        layers.add(LayerFactory.LstmLayer(inputSize,hiddenSize,useBias,init));
        return this;
    }

    public Layers addGRULayer(int inputSize, int hiddenSize, boolean useBias,Initializer init){
        layers.add(LayerFactory.GRULayer(inputSize,hiddenSize,useBias,init));
        return this;
    }

    public HashSet<Tensor> parameters(){
        HashSet <Tensor> params=new HashSet<>();
        for(Layer layer:layers){
            params.addAll(layer.parameters());
        }
        return params;
    }

    public HashSet<MemoryState> memoryList(){
        HashSet <MemoryState> memoryStateList =new HashSet<>();
        for(Layer layer:layers){
            memoryStateList.addAll(layer.memoryList());
        }
        return memoryStateList;
    }



}
