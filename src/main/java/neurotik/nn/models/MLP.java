package neurotik.nn.models;

import neurotik.data.TextVocabulary;
import neurotik.nn.Layer;
import neurotik.nn.Layers;
import neurotik.nn.Model;
import neurotik.nn.init.Initializer;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class MLP extends Model {
    public MLP(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        super(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
        System.out.println("Model: MLP | Parameters count: " + this.getParamsCount());
    }

    public MLP(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        super(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
        System.out.println("Model: MLP | Parameters count: " + this.getParamsCount());
    }

    @Override
    public Layers topology() {
        Layers l = new Layers();
        if (isTextBased) {
            l.addEmbedding(featuresOut, featuresIn);
        }
        l.addFlatten(contextLength - 1);
        for (int i = 0; i < hiddenLayers.length; i++) {
            if (i == 0) {
                l.addLinear(featuresIn * (contextLength - 1), hiddenLayers[i], useBias, init);
            } else {
                l.addLinear(hiddenLayers[i - 1], hiddenLayers[i], useBias, init);
            }
        }
        l.addLinear(hiddenLayers[hiddenLayers.length - 1], featuresOut, useBias, init);
        return l;
    }

    @Override
    public void generate(int samples) {
        if (vocabulary == null) {
            return;
        }
        Random random = new Random();
        for (int i = 0; i < samples; i++) {
            StringBuilder output = new StringBuilder();
            String context = ".".repeat(contextLength - 1);
            while (true) {
                Tensor predictions = forward(textInput(context));
                char nextChar = sampleNextChar(predictions, random);
                output.append(nextChar);
                if (nextChar == '.') {
                    break;
                }
                context = context.substring(1) + nextChar;
            }
            System.out.println(output);
        }
    }

    @Override
    public void generate(double[] initVals, int samples) {
        for (int i = 0; i < samples; i++) {
            Tensor inputs = numericInput(initVals);
            Tensor predictions = forward(inputs).compute();
            double nextValue = predictions.scalarAsDouble();
            double[] sub = DoubleStream.of(initVals).skip(1).limit(initVals.length - 1).toArray();
            System.arraycopy(sub, 0, initVals, 0, sub.length);
            initVals[initVals.length - 1] = nextValue;
            System.out.println(nextValue);
        }
    }

    private Tensor numericInput(double[] values) {
        double[] data = new double[values.length];
        System.arraycopy(values, 0, data, 0, values.length);
        return new Tensor(data, new int[]{values.length, 1, 1}, List.of(), "numeric prompt", DataType.FLOAT64);
    }
}
