package neurotik.nn.models;

import neurotik.encoding.Encoder;
import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.Layers;
import neurotik.tensor.MathHelper;
import neurotik.nn.Model;
import neurotik.data.DataSet;
import neurotik.encoding.TensorBatchEncoder;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class MLP extends Model{


    public MLP(int contextLength,int featuresIn, int [] hiddenLayers, int featuresOut,Encoder encoder,boolean useBias,Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,encoder,useBias,init);
        System.out.println("Model: MLP | Parameters count: "+this.getParamsCount());
    }

    public MLP(int contextLength, int featuresIn, int [] hiddenLayers, int featuresOut, boolean useBias,Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,useBias,init);
        System.out.println("Model: MLP | Parameters count: "+this.getParamsCount());
    }

    @Override
    public Layers topology(){
        Layers l=new Layers();
        l.addFlatten(contextLength-1);
        for (int i=0;i<hiddenLayers.length;i++){
            if (i==0)
                l.addLinear(featuresIn*(contextLength-1),hiddenLayers[i],useBias,init);
            else
                l.addLinear(hiddenLayers[i-1],hiddenLayers[i],useBias,init);
        }
        l.addLinear(hiddenLayers[hiddenLayers.length-1],featuresOut,useBias,init);
        return l;
    }



    @Override
    public void generate(int samples) {
        Random random=new Random();
        for (int i=0;i<samples;i++){
            String output="";
            String context="";

            Tensor probs;
            Tensor [] inputs;
            for (int j = 0; j < contextLength-1; j++) {
                context += ".";
            }



            while (true){
                inputs=this.encoder.encode(context);
                for (Layer layer:layers.layers){
                    inputs=layer.forward(inputs);
                }
                probs=inputs[0].softmax(1).compute();

                double[] probabilities = new double[probs.getDimensionAt(1)];
                System.arraycopy(probs.toDoubleArrayCopy(), 0, probabilities, 0, probabilities.length);
                int iChar=MathHelper.sampleFromMultinomial(1, probabilities,random)[0];
                char nextChar=encoder.decode(iChar);
                output=output+nextChar;
                if (iChar==0){
                    break;
                }
                context = context.substring(1) + nextChar;

            }

            System.out.println(output);
        }

    }


    @Override
    public void generate(double[] initVals,int samples){
        List<double[]> source=new ArrayList<>();
        source.add(initVals);
        double[] resultArray=new double[initVals.length];
        for (int i=0;i<samples;i++) {
            //clear all memories
            Tensor [] inputs= TensorBatchEncoder.encodeNumeric(new DataSet<>(source));
            for (Layer layer : layers.layers) {
                inputs = layer.forward(inputs);
            }
            double[] sub=DoubleStream.of(initVals).skip(1).limit(initVals.length-1).toArray();

            System.arraycopy(sub, 0, initVals, 0, sub.length);
            inputs[0].compute();
            initVals[initVals.length - 1] = inputs[0].scalarAsDouble();
            source.clear();
            source.add(initVals);
            System.out.println(inputs[0].scalarAsDouble());
        }
    }
}
