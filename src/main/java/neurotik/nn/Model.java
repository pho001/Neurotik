package neurotik.nn;

import neurotik.data.DataLoader;
import neurotik.encoding.Encoder;
import neurotik.encoding.EncoderFactory;
import neurotik.nn.init.Initializer;
import neurotik.nn.optim.Optimizer;
import neurotik.tensor.Tensor;

import java.util.Arrays;
import java.util.HashSet;
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


                //gradient reset
                Loss.resetGradients();
                //backward pass
                Loss.backward();
                updateParameters(optimizer, i);


                long endTime = System.nanoTime();
                durationInMillis += (endTime-startTime) / 1e6;
                cumLoss+=Loss.data[0][0]/ Losses.length;


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
            if (batch.getMask()!=null){
                this.setMask(batch.getMask());
            }

            inputs=batch.getInputs().padSortedSequences(".").encode(encoder);
            targets = batch.getTargets().padSortedSequences(".").encode(EncoderFactory.createOnehot(encoder.getVocab()));

        }
        else{
            inputs=batch.getInputs().encode(null);
            targets=batch.getTargets().encode(null);
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

                Losses[context] = inputs[context].categoricalEntropyLoss(targets[context].mapVec(IntStream.range(0, stepLengths[context]).toArray()));

            }
            else {
                Losses[context] = inputs[context].mse(targets[context]);
            }


            Losses[context].label="Loss "+context;


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
            cumLoss[j]=Loss.data[0][0];
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
            paramsCount+=p.rows*p.cols;
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


}
