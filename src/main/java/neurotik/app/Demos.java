package neurotik.app;

import neurotik.nn.optim.AdamOptimizer;
import neurotik.data.DataLoader;
import neurotik.encoding.Encoder;
import neurotik.encoding.EncoderFactory;
import neurotik.data.FileHandler;
import neurotik.nn.models.GRU;
import neurotik.nn.init.Initializer;
import neurotik.nn.init.InitializerFactory;
import neurotik.nn.models.LSTM;
import neurotik.nn.models.MLP;
import neurotik.nn.Model;
import neurotik.nn.ModelFactory;
import neurotik.data.NumDataSet;
import neurotik.nn.optim.Optimizer;
import neurotik.nn.optim.OptimizerFactory;
import neurotik.nn.models.RNN;
import neurotik.data.StringDataSet;
import neurotik.tensor.Tensor;

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
    static StringDataSet datas;
    static StringDataSet inputs;
    static StringDataSet outputs;
    static DataLoader<String> dl;

    static Encoder oneHotEnc;
    static Encoder vecEnc;
    static Optimizer opt;
    static Initializer init;
    static NumDataSet ns;
    static NumDataSet numinputs;
    static NumDataSet numoutputs;
    static DataLoader<double[]> numdl;
    static double [] vals;
    static List<double[]> sinvals;
    static StringDataSet ds;

    public static void runRnnTextDemo(){

        initText();
        datas=ds.setSequences(contextLength,false).sortDesc();
        inputs=datas.getSubSeq(0,-1);
        outputs=datas.getSubSeq(1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.RNN(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runLSTMTextDemo(){

        initText();
        datas=ds.setSequences(contextLength,false).sortDesc();
        inputs=datas.getSubSeq(0,-1);
        outputs=datas.getSubSeq(1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.LSTM(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runRnnNumbersDemo(){
        initNumbers();
        numinputs=ns.getSubSeq(0,-1);
        numoutputs=ns.getSubSeq(1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.RNN(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runLSTMNumbersDemo(){
        initNumbers();
        numinputs=ns.getSubSeq(0,-1);
        numoutputs=ns.getSubSeq(1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.LSTM(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runGRUNumbersDemo(){
        initNumbers();
        numinputs=ns.getSubSeq(0,-1);
        numoutputs=ns.getSubSeq(1,0);
        numdl=new DataLoader<>(numinputs,numoutputs);
        Model nn = ModelFactory.GRU(contextLength,1, hiddenLayers,1, false,init);
        nn.train(numdl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(numdl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(DoubleStream.of(vals).skip(0).limit(contextLength-1).toArray(),100);

    }

    public static void runGRUTextDemo(){

        initText();
        datas=ds.setSequences(contextLength,false).sortDesc();
        inputs=datas.getSubSeq(0,-1);
        outputs=datas.getSubSeq(1,0);
        dl=new DataLoader<>(inputs,outputs);
        Model nn = ModelFactory.GRU(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.setMask(null);
        nn.generate(100);
    }

    public static void runMLPTextDemo(){
        initText();
        datas=ds.setSequences(contextLength,true);
        inputs=datas.getSubSeq(0,-1);
        outputs=datas.getSubSeq(-1);
        dl=new DataLoader<>(inputs,outputs);
        Model nn=ModelFactory.MLP(contextLength,embeddingSize, hiddenLayers,vecEnc.getVocab().size(), vecEnc,false,init);
        nn.train(dl.trainingSet(), batchSize, epochs, opt);
        nn.getSetLoss(dl.testSetData(),batchSize);
        nn.generate(100);
    }

    public static void runMLPNumbersDemo(){
        initNumbers();
        numinputs=ns.getSubSeq(0,-1);
        numoutputs=ns.getSubSeq(contextLength-1,0);
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
        FileHandler fh = new FileHandler("data/names.txt");
        List<String> set = fh.ReadFileLines();
        Collections.shuffle(set);
        ds = new StringDataSet(set);
        alphabet = "." + ds.uniqueCharacters();


        oneHotEnc = EncoderFactory.createOnehot(alphabet);
        vecEnc =  EncoderFactory.createVecEmb(alphabet, embeddingSize);
        opt = OptimizerFactory.AdamOptimizer(0.01);
        init=InitializerFactory.kaimingInit();
    }

    private static void initNumbers(){
        List<double[]> sinvals=new ArrayList<>();
        vals=sinWave(10000,0.1,0.2);
        sinvals.add(vals);
        ns = new NumDataSet(sinvals).setSequences(contextLength,false);

        opt = OptimizerFactory.AdamOptimizer(0.01);
        init=InitializerFactory.kaimingInit();
    }


}
