package neurotik.app;

import neurotik.data.DataLoader;
import neurotik.data.DataSet;
import neurotik.data.NumericDatasets;
import neurotik.data.NumericSequenceCollator;
import neurotik.data.SequenceBatch;
import neurotik.data.SupervisedDataSplit;
import neurotik.data.SupervisedDataset;
import neurotik.data.TextDatasets;
import neurotik.data.TextFileReader;
import neurotik.data.TextSequenceCollator;
import neurotik.data.TextVocabulary;
import neurotik.nn.Model;
import neurotik.nn.ModelFactory;
import neurotik.nn.init.Initializer;
import neurotik.nn.init.InitializerFactory;
import neurotik.nn.optim.Optimizer;
import neurotik.nn.optim.OptimizerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class Demos {
    static int contextLength = 50;
    static int embeddingSize = 10;
    static int batchSize = 64;
    static int epochs = 3;
    static int[] hiddenLayers = {100};

    static TextVocabulary vocabulary;
    static Optimizer opt;
    static Initializer init;
    static double[] vals;
    static DataSet<String> ds;

    public static void runRnnTextDemo() {
        initText();
        SupervisedDataSplit<String, String> split = TextDatasets
                .sequenceNextChar(ds, contextLength, ".")
                .shuffle(new Random())
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.RNN(contextLength, embeddingSize, hiddenLayers, vocabulary, false, init);
        trainAndEvaluate(nn, textLoader(split.train()), textLoader(split.test()));
        nn.generate(100);
    }

    public static void runLSTMTextDemo() {
        initText();
        SupervisedDataSplit<String, String> split = TextDatasets
                .sequenceNextChar(ds, contextLength, ".")
                .shuffle(new Random())
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.LSTM(contextLength, embeddingSize, hiddenLayers, vocabulary, false, init);
        trainAndEvaluate(nn, textLoader(split.train()), textLoader(split.test()));
        nn.generate(100);
    }

    public static void runGRUTextDemo() {
        initText();
        SupervisedDataSplit<String, String> split = TextDatasets
                .sequenceNextChar(ds, contextLength, ".")
                .shuffle(new Random())
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.GRU(contextLength, embeddingSize, hiddenLayers, vocabulary, false, init);
        trainAndEvaluate(nn, textLoader(split.train()), textLoader(split.test()));
        nn.generate(100);
    }

    public static void runMLPTextDemo() {
        initText();
        SupervisedDataSplit<String, String> split = TextDatasets
                .fixedNextChar(ds, contextLength, ".")
                .shuffle(new Random())
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.MLP(contextLength, embeddingSize, hiddenLayers, vocabulary, false, init);
        trainAndEvaluate(nn, textLoader(split.train()), textLoader(split.test()));
        nn.generate(100);
    }

    public static void runRnnNumbersDemo() {
        initNumbers();
        SupervisedDataSplit<double[], double[]> split = NumericDatasets
                .sequencePrediction(new DataSet<>(List.of(vals)), contextLength)
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.RNN(contextLength, 1, hiddenLayers, 1, false, init);
        trainAndEvaluate(nn, numericLoader(split.train()), numericLoader(split.test()));
        nn.generate(seedValues(), 100);
    }

    public static void runLSTMNumbersDemo() {
        initNumbers();
        SupervisedDataSplit<double[], double[]> split = NumericDatasets
                .sequencePrediction(new DataSet<>(List.of(vals)), contextLength)
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.LSTM(contextLength, 1, hiddenLayers, 1, false, init);
        trainAndEvaluate(nn, numericLoader(split.train()), numericLoader(split.test()));
        nn.generate(seedValues(), 100);
    }

    public static void runGRUNumbersDemo() {
        initNumbers();
        SupervisedDataSplit<double[], double[]> split = NumericDatasets
                .sequencePrediction(new DataSet<>(List.of(vals)), contextLength)
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.GRU(contextLength, 1, hiddenLayers, 1, false, init);
        trainAndEvaluate(nn, numericLoader(split.train()), numericLoader(split.test()));
        nn.generate(seedValues(), 100);
    }

    public static void runMLPNumbersDemo() {
        initNumbers();
        SupervisedDataSplit<double[], double[]> split = NumericDatasets
                .fixedPrediction(new DataSet<>(List.of(vals)), contextLength)
                .split(0.8, 0.1, 0.1);
        Model nn = ModelFactory.MLP(contextLength, 1, hiddenLayers, 1, false, init);
        trainAndEvaluate(nn, numericLoader(split.train()), numericLoader(split.test()));
        nn.generate(seedValues(), 100);
    }

    public static double[] sinWave(int points, double step, double noiseLevel) {
        double[] sinValues = new double[points];
        Random random = new Random();
        for (int i = 0; i < points; i++) {
            double noise = (random.nextDouble() * 2 - 1) * noiseLevel;
            sinValues[i] = Math.sin(i * step) + noise;
        }
        return sinValues;
    }

    private static void initText() {
        List<String> set = new ArrayList<>(TextFileReader.readLines("data/names.txt").data());
        Collections.shuffle(set);
        ds = new DataSet<>(set);
        vocabulary = TextVocabulary.from(ds, ".");
        opt = OptimizerFactory.AdamOptimizer(0.01);
        init = InitializerFactory.kaimingInit();
    }

    private static void initNumbers() {
        vals = sinWave(10000, 0.1, 0.2);
        opt = OptimizerFactory.AdamOptimizer(0.01);
        init = InitializerFactory.kaimingInit();
    }

    private static DataLoader<SequenceBatch> textLoader(SupervisedDataset<String, String> dataset) {
        return DataLoader.from(dataset)
                .batchSize(batchSize)
                .collator(TextSequenceCollator.indices(vocabulary, "."));
    }

    private static DataLoader<SequenceBatch> numericLoader(SupervisedDataset<double[], double[]> dataset) {
        return DataLoader.from(dataset)
                .batchSize(batchSize)
                .collator(new NumericSequenceCollator());
    }

    private static void trainAndEvaluate(Model model, DataLoader<SequenceBatch> train, DataLoader<SequenceBatch> test) {
        model.train(train, epochs, opt);
        model.getSetLoss(test);
        model.setMask(null);
    }

    private static double[] seedValues() {
        return DoubleStream.of(vals).limit(contextLength - 1).toArray();
    }
}
