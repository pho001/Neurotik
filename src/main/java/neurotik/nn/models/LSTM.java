package neurotik.nn.models;

import neurotik.data.TextVocabulary;
import neurotik.nn.Layers;
import neurotik.nn.Model;
import neurotik.nn.init.Initializer;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Random;

public class LSTM extends Model {
    public LSTM(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        super(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
        System.out.println("Model: LSTM | Parameters count: " + this.getParamsCount());
    }

    public LSTM(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        super(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
        System.out.println("Model: LSTM | Parameters count: " + this.getParamsCount());
    }

    @Override
    public void generate(int samples) {
        generate(".", samples);
    }

    @Override
    public void generate(String prompt, int samples) {
        if (vocabulary == null) {
            return;
        }
        Random random = new Random();
        this.setLearningMode(false);
        for (int i = 0; i < samples; i++) {
            this.resetMemoryStates();
            StringBuilder output = new StringBuilder();
            String context = prompt;
            while (true) {
                Tensor predictions = forward(textInput(context));
                char nextChar = sampleNextChar(predictions, random);
                output.append(nextChar);
                if (nextChar == '.') {
                    break;
                }
                context = "" + nextChar;
            }
            System.out.println(output);
        }
    }

    @Override
    public void generate(double[] initVals, int samples) {
        this.resetMemoryStates();
        for (int i = 0; i < samples; i++) {
            Tensor predictions = forward(new Tensor(new double[]{initVals[initVals.length - 1]}, new int[]{1, 1, 1}, List.of(), "numeric prompt", DataType.FLOAT64)).compute();
            double nextValue = predictions.select(0, predictions.getDimensionAt(0) - 1).scalarAsDouble();
            initVals[initVals.length - 1] = nextValue;
            System.out.println(nextValue);
        }
    }

    @Override
    public Layers topology() {
        Layers l = new Layers();
        if (isTextBased) {
            l.addEmbedding(featuresOut, featuresIn);
        }
        for (int i = 0; i < hiddenLayers.length; i++) {
            if (i == 0) {
                l.addLstmLayer(featuresIn, hiddenLayers[i], useBias, init);
            } else {
                l.addLstmLayer(hiddenLayers[i - 1], hiddenLayers[i], useBias, init);
            }
        }
        l.addLinear(hiddenLayers[hiddenLayers.length - 1], featuresOut, useBias, init);
        return l;
    }
}
