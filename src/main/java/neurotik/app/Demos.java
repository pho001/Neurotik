package neurotik.app;

import neurotik.nn.optim.AdamOptimizer;
import neurotik.data.DataLoader;
import neurotik.data.DataSet;
import neurotik.encoding.Encoder;
import neurotik.encoding.EncoderFactory;
import neurotik.data.TextDataTransforms;
import neurotik.data.TextFileReader;
import neurotik.nn.models.GRU;
import neurotik.nn.init.Initializer;
import neurotik.nn.init.InitializerFactory;
import neurotik.nn.models.LSTM;
import neurotik.nn.models.MLP;
import neurotik.nn.Model;
import neurotik.nn.ModelFactory;
import neurotik.data.NumericDataTransforms;
import neurotik.nn.optim.Optimizer;
import neurotik.nn.optim.OptimizerFactory;
import neurotik.nn.models.RNN;
import tensor.Tensor;

import java.util.*;
import java.util.stream.DoubleStream;

public class Demos {

    static int contextLength=50;
    static int embeddingSize=10;
    static int batchSize=64;

    static int epochs=3;
    static int samples;
    static int[] hiddenLayers={100};
    static HashSet<Tensor> params;

    static String alphabet;
    static DataSet<String> datas;
    static DataSet<String> inputs;
    static DataSet<String> outputs;
    static DataLoader<String> dl;

    static Encoder oneHotEnc;
    static Encoder vecEnc;
    static Optimizer opt;
    static Initializer init;
    static DataSet<double[]> ns;
    static DataSet<double[]> numinputs;
    static DataSet<double[]> numoutputs;
    static DataLoader<double[]> numdl;
    static double [] vals;
    static List<double[]> sinvals;
    static DataSet<String> ds;

    public static void runRnnTextDemo(){

        initText();
        datas=TextDataTransforms.sortByLengthDesc(TextDataTransforms.charWindows(ds, contextLength, false));
        inputs=TextDataTransforms.sliceOffsets(datas, 0,-1);
        outputs=TextDataTransforms.sliceOffsets(datas, 1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.RNN(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runLSTMTextDemo(){

        initText();
        datas=TextDataTransforms.sortByLengthDesc(TextDataTransforms.charWindows(ds, contextLength, false));
        inputs=TextDataTransforms.sliceOffsets(datas, 0,-1);
        outputs=TextDataTransforms.sliceOffsets(datas, 1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.LSTM(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runRnnNumbersDemo(){
        initNumbers();
        numinputs=NumericDataTransforms.sliceOffsets(ns, 0,-1);
        numoutputs=NumericDataTransforms.sliceOffsets(ns, 1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.RNN(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runLSTMNumbersDemo(){
        initNumbers();
        numinputs=NumericDataTransforms.sliceOffsets(ns, 0,-1);
        numoutputs=NumericDataTransforms.sliceOffsets(ns, 1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.LSTM(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runGRUNumbersDemo(){
        initNumbers();
        numinputs=NumericDataTransforms.sliceOffsets(ns, 0,-1);
        numoutputs=NumericDataTransforms.sliceOffsets(ns, 1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.GRU(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runGRUTextDemo(){

        initText();
        datas=TextDataTransforms.sortByLengthDesc(TextDataTransforms.charWindows(ds, contextLength, false));
        inputs=TextDataTransforms.sliceOffsets(datas, 0,-1);
        outputs=TextDataTransforms.sliceOffsets(datas, 1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.GRU(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runMLPTextDemo(){
        initText();
        datas=TextDataTransforms.charWindows(ds, contextLength,true);
        inputs=TextDataTransforms.sliceOffsets(datas, 0,-1);
        outputs=TextDataTransforms.tail(datas, -1);
        dl=new DataLoader<>(inputs,outputs);
        Model nn=ModelFactory.MLP(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.generate(100);
    }

    public static void runMLPNumbersDemo(){
        initNumbers();
        numinputs=NumericDataTransforms.sliceOffsets(ns, 0,-1);
        numoutputs=NumericDataTransforms.sliceOffsets(ns, contextLength-1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.MLP(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);
    }

    public static double[] sinWave(int points, double step,double noiseLevel){

        double[] sinValues = new double[points];

        Random random = new Random();
        for (int i = 0; i < points; i++) {
            double noise = (random.nextDouble() * 2 - 1) * noiseLevel;
            sinValues[i] = Math.sin(i*step)+noise;
        }

        return sinValues;
    }

    private static void initText(){


        params = new HashSet<>();
        List<String> set = new ArrayList<>(TextFileReader.readLines("data/names.txt").data());
        Collections.shuffle(set);
        ds = new DataSet<>(set);
        alphabet = "." + TextDataTransforms.uniqueCharacters(ds);


        oneHotEnc = EncoderFactory.createOnehot(alphabet);
        vecEnc =  EncoderFactory.createVecEmb(alphabet, embeddingSize);
        opt = OptimizerFactory.AdamOptimizer(0.01);
        init=InitializerFactory.kaimingInit();
    }

    private static void initNumbers(){
        List<double[]> sinvals=new ArrayList<>();
        vals=sinWave(10000,0.1,0.2);
        sinvals.add(vals);
        ns = NumericDataTransforms.windows(new DataSet<>(sinvals), contextLength);

        opt = OptimizerFactory.AdamOptimizer(0.01);
        init=InitializerFactory.kaimingInit();
    }


}
