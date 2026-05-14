package neurotik.nn;

import neurotik.data.TextVocabulary;
import neurotik.nn.init.Initializer;
import neurotik.nn.models.GRU;
import neurotik.nn.models.LSTM;
import neurotik.nn.models.MLP;
import neurotik.nn.models.RNN;

public class ModelFactory {
    public static Model MLP(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        return new MLP(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
    }

    public static Model MLP(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        return new MLP(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
    }

    public static Model RNN(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        return new RNN(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
    }

    public static Model RNN(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        return new RNN(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
    }

    public static Model LSTM(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        return new LSTM(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
    }

    public static Model LSTM(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        return new LSTM(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
    }

    public static Model GRU(int contextLength, int embeddingSize, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        return new GRU(contextLength, embeddingSize, hiddenLayers, vocabulary, useBias, init);
    }

    public static Model GRU(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        return new GRU(contextLength, featuresIn, hiddenLayers, featuresOut, useBias, init);
    }

}
