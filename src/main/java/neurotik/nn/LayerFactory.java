package neurotik.nn;

import neurotik.nn.activation.Activation;
import neurotik.nn.layers.ActivationLayer;
import neurotik.nn.layers.FlattenLayer;
import neurotik.nn.layers.GRULayer;
import neurotik.nn.init.Initializer;
import neurotik.nn.layers.LinearLayer;
import neurotik.nn.layers.LstmLayer;
import neurotik.nn.layers.OneDirectionalRNNLayer;

import java.util.List;

public class LayerFactory {
    public static Layer LinearLayer(int features_in, int features_out, boolean useBias, Initializer init) {
        return new LinearLayer(features_in,features_out,useBias,init);
    }
    public static Layer FlattenLayer(int factor) {
        return new FlattenLayer(factor);
    }

    public static Layer ActivationLayer(Activation activation) {
        return new ActivationLayer(activation);
    }

    public static Layer OneDirectionalRNN(int inputSize, int hiddenSize,boolean useBias,Initializer init) {
        return new OneDirectionalRNNLayer(inputSize,hiddenSize,useBias,init);
    }
    public static Layer LstmLayer(int inputSize, int hiddenSize,boolean useBias,Initializer init) {
        return new LstmLayer(inputSize,hiddenSize,useBias,init);
    }
    public static Layer GRULayer(int inputSize, int hiddenSize,boolean useBias,Initializer init) {
        return new GRULayer(inputSize,hiddenSize,useBias,init);
    }
}
