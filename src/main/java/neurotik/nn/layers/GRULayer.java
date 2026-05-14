package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.tensor.MathHelper;
import neurotik.nn.MemoryState;
import neurotik.tensor.Tensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

public class GRULayer extends Layer{

    boolean useBias=false;

    int step;


    Tensor bias_h;

    Tensor Wu;
    Tensor Wr;
    Tensor Wh;




    int hiddenSize;
    int inputSize;
    MemoryState lastMemoryState;



    Tensor [] out;



    public GRULayer(int inputSize, int hiddenSize,boolean useBias,Initializer init){
        this.useBias=useBias;
        this.step=0;

        if (this.useBias==true) {
            this.bias_h = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_h").zeros();
        }

        this.Wu=init.init(inputSize+hiddenSize,hiddenSize);
        this.Wr=init.init(inputSize+hiddenSize,hiddenSize);
        this.Wh=init.init(inputSize+hiddenSize,hiddenSize);

        this.hiddenSize=hiddenSize;
        this.inputSize=inputSize;


        lastMemoryState=new MemoryState();


    }



    @Override
    public Tensor[] forward(Tensor [] input){
        out =new Tensor[input.length];
        int[] packedSizes =null;
        if (mask!=null) {
            packedSizes = getPackedBatchesSizes();
        }

        //if hidden state wasn't initialized yet
        if (lastMemoryState.get()==null){
            lastMemoryState=new MemoryState(new Tensor(input[0].rows,hiddenSize,new HashSet<>(),"h(-1)"));
        }
        //let's calculate for each time step
        for (int step=0;step<input.length;step++){
            input[step].label="data("+step+")";
            if (mask!=null) {
                int[] indexes= IntStream.rangeClosed(0, packedSizes[step]-1).toArray();
                out[step] = gruCell(input[step].mapVec(indexes), lastMemoryState.get().mapVec(indexes));
            }
            else {
                out[step] = gruCell(input[step], lastMemoryState.get());
            }
            lastMemoryState.get().label="h("+step+")";
        }


        return out;
    }

    public Tensor gruCell(Tensor input,Tensor hprev){
        Tensor updateGate=null;
        Tensor resetGate=null;
        Tensor candidateHidState=null;



        Tensor z= input.concatRight(hprev);

        if (useBias) {
            updateGate=z.mul(Wu).addb(bias_h).sigmoid();
            resetGate=z.mul(Wr).addb(bias_h).sigmoid();
            candidateHidState=input.concatRight(resetGate.hadamard(hprev)).mul(Wh).addb(bias_h).tanh();
        }
        else {
            updateGate=z.mul(Wu).sigmoid();
            resetGate=z.mul(Wr).sigmoid();
            candidateHidState=input.concatRight(resetGate.hadamard(hprev)).mul(Wh).tanh();
        }


        Tensor ones=new Tensor(MathHelper.ones(updateGate.rows,updateGate.cols),new HashSet<>(),"ones");
        Tensor hidden=ones.sub(updateGate).hadamard(hprev).add(updateGate.hadamard(candidateHidState));
        lastMemoryState= new MemoryState(hidden);
        return hidden;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();

        params.add(this.Wu);
        params.add(this.Wr);
        params.add(Wh);

        if (useBias)
            params.add(this.bias_h);
        return params;
    }



    @Override
    public void setLearningMode(boolean learningMode){
        this.learningMode=learningMode;

    }

    @Override
    public HashSet<MemoryState> memoryList() {
        HashSet<MemoryState> memoryState =new HashSet<>();
        memoryState.add(this.lastMemoryState);
        return memoryState;
    }

    @Override
    public void initParameters(Initializer init) {

        Wh=init.init(inputSize+hiddenSize,hiddenSize);
        Wh.label="Wh";
        Wu=init.init(inputSize+hiddenSize,hiddenSize);
        Wu.label="Wu";
        Wr=init.init(inputSize+hiddenSize,hiddenSize);
        Wr.label="Wr";
    }

    private int[] getPackedBatchesSizes(){

        int min= Arrays.stream(mask).min().orElseThrow(() -> new IllegalArgumentException("The array must not be null or empty"));
        int max=Arrays.stream(mask).max().orElseThrow(() -> new IllegalArgumentException("The array must not be null or empty"));
        int [] lengths=new int[max];
        for (int i=0;i<max;i++){
            final int f=i;
            int count= (int) Arrays.stream(mask).filter(value->value >f).count();
            lengths[i]=count;
        }
        return lengths;
    }

}
