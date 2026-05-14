package neurotik.nn;

import neurotik.data.DataLoader;
import neurotik.data.DataSet;
import neurotik.data.TextDataTransforms;
import neurotik.encoding.Encoder;
import neurotik.encoding.EncoderFactory;
import neurotik.encoding.TensorBatchEncoder;
import neurotik.nn.init.Initializer;
import neurotik.nn.optim.Optimizer;
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.OptionalDouble;
import java.util.stream.IntStream;

public abstract class Model {
    protected boolean learningMode;

    //last layer must be output layer
    protected int [] hiddenLayers;
    protected Layers layers;
    protected int contextLength;
    protected boolean useBias;
    protected Encoder encoder;
    protected boolean isTextBased;
    protected int featuresIn;
    protected int featuresOut;
    protected Initializer init;





    public Model(int contextLength,int featuresIn, int [] hiddenLayers, int featuresOut,Encoder encoder,boolean useBias,Initializer init){
        //Initial state of model is learning mode
        this.contextLength=contextLength;
        this.hiddenLayers=hiddenLayers;
        this.encoder=encoder;
        this.useBias= useBias;
        this.featuresIn=featuresIn;
        this.featuresOut=featuresOut;
        this.init=init;
        this.layers=topology();
        this.learningMode=false;
        this.isTextBased=true;



    }

    public Model(int contextLength,int featuresIn, int [] hiddenLayers, int featuresOut,boolean useBias,Initializer init){
        //Initial state of model is learning mode
        this.contextLength=contextLength;
        this.hiddenLayers=hiddenLayers;
        this.useBias= useBias;
        this.featuresIn=featuresIn;
        this.featuresOut=featuresOut;
        this.init=init;
        this.layers=topology();
        this.learningMode=false;
        this.isTextBased=false;

    }
    public HashSet<Tensor> parameters(){
        HashSet <Tensor> params=new HashSet<>();
        for(Layer layer:layers.layers){
            params.addAll(layer.parameters());
        }
        if (encoder!=null) {
            params.addAll(encoder.parameters());
        }

        return params;
    }

    public HashSet<MemoryState> memoryList(){
        HashSet <MemoryState> memoryStateList =new HashSet<>();
        for(Layer layer:layers.layers){
            memoryStateList.addAll(layer.memoryList());
        }
        return memoryStateList;
    }

    //public abstract <T> void train(DataLoader<T> ts, int batchSize,int epochs,Optimizer optimizer);

    public <T> void train(DataLoader<T> ds, int batchSize, int epochs,Optimizer optimizer) {
        System.out.println("Parameters count: "+this.getParamsCount());
        double durationInMillis=0;
        double cumLoss=0;
        int lines=100;
        int iterations=ds.getSetSize()/batchSize;
        this.setLearningMode(true);
        for (int i=0;i<epochs;i++){

            for (int j=0;j<iterations;j++) {
                long startTime = System.nanoTime();
                this.resetMemoryStates();
                int startIndex = j * batchSize;
                DataLoader<T> batch = ds.getBatch(batchSize, startIndex);
                Tensor[] Losses=getLoss(batch);
                Tensor Loss=null;
                for (int l=0;l<Losses.length;l++){
                    if (l==0){
                        Loss=Losses[0];
                    }
                    else {
                        Loss = Loss.add(Losses[l]);
                    }
                }


                for (Tensor parameter : parameters()) {
                    TensorInternalAccess.clearGradient(parameter);
                }
                Loss.compute(CompileMode.TRAINING);
                updateParameters(optimizer, i);


                long endTime = System.nanoTime();
                durationInMillis += (endTime-startTime) / 1e6;
                cumLoss+=Loss.scalarAsDouble()/ Losses.length;


                if (j%lines==0 && j>0) {
                    System.out.println("Epoch: " + i + " || Iteration " + j + "/" + iterations + ": " + cumLoss / lines + " || Timesteps: " + Losses.length + " || Avg. duration : " + durationInMillis / lines + " ms");
                    durationInMillis = 0;
                    cumLoss = 0;
                }
                else if (j==iterations-1){
                    System.out.println("Epoch: " + i + " || Iteration " + (j+1) + "/" + iterations + ": " + cumLoss / (j%lines) + " || Timesteps: " + Losses.length + " || Avg. duration : " + durationInMillis / (j%lines) + " ms");
                    durationInMillis = 0;
                    cumLoss = 0;
                }

            }

        }
    }

    public <T> Tensor[] getLoss(DataLoader<T> batch){

        Tensor [] inputs=null;
        Tensor [] targets=null;
        if(encoder!=null){
            DataLoader<String> textBatch = strings(batch).sortByInputs(Comparator.comparingInt(String::length).reversed());
            DataSet<String> inputStrings = textBatch.getInputs();
            DataSet<String> targetStrings = textBatch.getTargets();
            this.setMask(TextDataTransforms.lengths(inputStrings));

            inputs = TensorBatchEncoder.encodeText(TextDataTransforms.padToMaxLength(inputStrings, "."), encoder);
            targets = TensorBatchEncoder.encodeText(
                    TextDataTransforms.padToMaxLength(targetStrings, "."),
                    EncoderFactory.createOnehot(encoder.getVocab()));

        }
        else{
            inputs = TensorBatchEncoder.encodeNumeric(numbers(batch.getInputs()));
            targets = TensorBatchEncoder.encodeNumeric(numbers(batch.getTargets()));
        }
        //---------- Forward pass-----------
        for (Layer layer:layers.layers){
            inputs=layer.forward(inputs);
        }

        Tensor Loss=null;
        Tensor [] Losses=new Tensor[inputs.length];
        for (int context=0;context<Losses.length;context++){

            if (isTextBased) {
                int [] stepLengths=batch.getPackedBatchesSizes();

                int[] indexes = IntStream.range(0, stepLengths[context]).toArray();
                Tensor indexTensor = new Tensor(indexes, new int[]{indexes.length}, List.of(), "target indexes", DataType.INT32);
                Losses[context] = inputs[context].crossEntropyLoss(targets[context].gatherAxis(indexTensor, 0), 1);

            }
            else {
                Losses[context] = inputs[context].sub(targets[context]).pow(2).mean();
            }


            Losses[context].setLabel("Loss "+context);


        }
        return Losses;
    }

    public <T> void getSetLoss(DataLoader<T> dl,int batchSize){
        int iterations=dl.getSetSize()/batchSize;
        double [] cumLoss=new double[iterations];
        for (int j = 0; j < iterations; j++) {
            this.resetMemoryStates();
            Tensor[] Losses = getLoss(dl.getBatch(batchSize,j*batchSize));
            Tensor Loss = null;
            for (int l = 0; l < Losses.length; l++) {
                if (l == 0) {
                    Loss = Losses[0];
                } else {
                    Loss = Loss.add(Losses[l]);
                }
            }
            Loss.compute();
            cumLoss[j]=Loss.scalarAsDouble();
        }
        OptionalDouble avg=Arrays.stream(cumLoss).average();
        if (avg.isPresent())
            System.out.println("Test set loss: " + avg.getAsDouble());

    }

    public abstract void generate(int samples);

    public void generate(String prompt, int samples){

    }

    public void generate(double[] initValues, int samples){

    }

    public void setLearningMode(boolean learningMode) {
        this.learningMode=learningMode;
        for(Layer layer:layers.layers){
            layer.setLearningMode(learningMode);
        }
    }

    public void updateParameters(Optimizer opt,int epoch){
        HashSet<Tensor> parameters = parameters();
        for (Tensor p:parameters){
            opt.update(p,epoch+1);
        }
    }


    public void resetMemoryStates(){
        HashSet <MemoryState> memoryStateList = memoryList();
        for (MemoryState memoryState : memoryStateList){
            if (memoryState !=null)
                memoryState.reset();
        }
    }

    public abstract Layers topology();




    public int getParamsCount(){
        int paramsCount=0;
        HashSet<Tensor> parameters = parameters();
        for (Tensor p:parameters){
            int tensorParams = 1;
            for (int dimension : p.getShape()) {
                tensorParams *= dimension;
            }
            paramsCount += tensorParams;
        }
        return paramsCount;
    }

    public void initParameters(Initializer initializer){
        for (Layer l:layers.layers){
            l.initParameters(initializer);
        }
    }

    public void setMask(int [] mask){
        for (Layer l:layers.layers){
            l.setMask(mask);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> DataLoader<String> strings(DataLoader<T> dataLoader) {
        return (DataLoader<String>) dataLoader;
    }

    @SuppressWarnings("unchecked")
    private static <T> DataSet<double[]> numbers(DataSet<T> dataSet) {
        return (DataSet<double[]>) dataSet;
    }

}
