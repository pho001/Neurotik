package neurotik.nn.models;

import neurotik.nn.activation.ActivationFactory;
import neurotik.encoding.Encoder;
import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.Layers;
import neurotik.tensor.MathHelper;
import neurotik.nn.Model;
import neurotik.data.NumDataSet;
import neurotik.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class RNN extends Model{

    public RNN(int contextLength,int featuresIn, int [] hiddenLayers, int featuresOut,Encoder encoder,boolean useBias,Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,encoder,useBias,init);
        System.out.println("Model: RNN | Parameters count: "+this.getParamsCount());
    }

    public RNN(int contextLength, int featuresIn, int [] hiddenLayers, int featuresOut, boolean useBias,Initializer init){
        super(contextLength,featuresIn,hiddenLayers,featuresOut,useBias,init);
        System.out.println("Model: RNN | Parameters count: "+this.getParamsCount());
    }



    @Override
    public void generate(int samples) {
        generate(".",samples);
    }

    @Override
    public void generate(String prompt,int samples){
        Random random=new Random();
        Tensor [] inputs;
        this.setLearningMode(false);
        //let's set layers to predictive mode



        for (int i=0;i<samples;i++) {
            //generative mode
            this.resetMemoryStates();
            String output = "";
            String context=prompt;


            while (true) {

                inputs = this.encoder.encode(context);
                for (Layer layer : layers.layers) {
                    inputs = layer.forward(inputs);
                }

                Tensor probs = inputs[inputs.length-1].softMax();
                int iChar = MathHelper.sampleFromMultinomial(1, probs.data[0], random)[0];
                char nextChar = encoder.decode(iChar);
                output = output + nextChar;
                if (iChar == 0) {
                    break;
                }
                //context = context.substring(1) + nextChar;
                context = "" + nextChar;
                //context = context + nextChar;
            }
            System.out.println(output);

        }
    }

    @Override
    public void generate(double[] initVals,int samples){
        List<double[]> source=new ArrayList<>();
        source.add(initVals);
        double[] resultArray=new double[initVals.length];
        this.resetMemoryStates();
        NumDataSet ns = new NumDataSet(source);
        for (int i=0;i<samples;i++) {
            //clear all memories
            ns=new NumDataSet(source);
            Tensor [] inputs=ns.encode(null);
            for (Layer layer : layers.layers) {
                inputs = layer.forward(inputs);
            }
            double[] sub=DoubleStream.of(initVals).skip(1).limit(initVals.length-1).toArray();

            System.arraycopy(sub, 0, initVals, 0, sub.length);
            initVals[initVals.length - 1] = inputs[inputs.length-1].data[0][0];
            source.clear();
            //source.add(initVals);
            source.add(new double[]{inputs[inputs.length-1].data[0][0]});
            System.out.println(inputs[inputs.length-1].data[0][0]);
        }
    }


    @Override
    public Layers topology() {
        Layers l=new Layers();

        for (int i = 0; i < hiddenLayers.length; i++){
            if (i==0)
                l.addOneDirectionalRNN(featuresIn, hiddenLayers[i], useBias,init);
            else
                l.addOneDirectionalRNN(hiddenLayers[i-1],hiddenLayers[i],useBias,init);
        }

        //l.addActivation(ActivationFactory.ReLU());
        l.addLinear(hiddenLayers[hiddenLayers.length-1],featuresOut,useBias,init);
        return l;
    }







}
