package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.MemoryState;
import neurotik.tensor.Tensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class OneDirectionalRNNLayer extends Layer{

    boolean useBias=false;

    int step;

    Tensor Whh;
    Tensor bias_h;

    Tensor Wx;
    Tensor Wh;




    int hiddenSize;
    int inputSize;
    MemoryState lastMemoryState;



    Tensor [] out;



    public OneDirectionalRNNLayer(int inputSize, int hiddenSize,boolean useBias,Initializer init){
        this.useBias=useBias;
        this.step=0;
        this.Whh=init.init(hiddenSize,hiddenSize);
        Whh.label="Whh";
        this.Wx=init.init(inputSize,hiddenSize);
        Wx.label="Wx";

        if (this.useBias==true) {
            this.bias_h = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_h").zeros();
        }

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
                out[step] = rnnCell(input[step].mapVec(indexes), lastMemoryState.get().mapVec(indexes));
            }
            else {
                out[step] = rnnCell(input[step], lastMemoryState.get());
            }
            lastMemoryState.get().label="h("+step+")";
        }


        return out;
    }

    public Tensor rnnCell(Tensor input,Tensor hprev){
        Tensor hidden=null;


        Tensor z= input.concatRight(hprev);

        if (useBias)
            hidden=z.mul(Wh).addb(bias_h).tanh();
        else
            hidden=z.mul(Wh).tanh();


        /*
        if (useBias)
            hidden=(input.mul(Wx).add(hprev.mul(Whh)).addb(bias_h)).tanh();
        else
            hidden=(input.mul(Wx).add(hprev.mul(Whh))).tanh();
        */


        lastMemoryState= new MemoryState(hidden);
        return hidden;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        /*
        params.add(this.Wx);
        params.add(this.Whh);
        */

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
        /*
        Whh=init.init(hiddenSize,hiddenSize);
        Whh.label="Whh";
        Wx=init.init(inputSize,hiddenSize);
        Wx.label="Wx";
        */
        Wh=init.init(inputSize+hiddenSize,hiddenSize);
        Wh.label="Wh";

    }

    private int[] getPackedBatchesSizes(){

        int min=Arrays.stream(mask).min().orElseThrow(() -> new IllegalArgumentException("The array must not be null or empty"));
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
