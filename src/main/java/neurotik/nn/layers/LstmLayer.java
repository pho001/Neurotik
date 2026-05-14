package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.MemoryState;
import neurotik.tensor.Tensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

public class LstmLayer extends Layer{

    boolean useBias=false;

    int step;


    Tensor Bi;
    Tensor Bf;

    Tensor Bo;
    Tensor Bc;



    Tensor Wf;
    Tensor Wi;

    Tensor Wo;
    Tensor Wc;



    int inputSize;
    int hiddenSize;

    MemoryState hiddenState;
    MemoryState cellState;




    Tensor [] hidden;
    Tensor [] out;

    public LstmLayer(int inputSize, int hiddenSize,boolean useBias,Initializer init){
        this.useBias=useBias;
        this.step=0;

        this.Wf=init.init(hiddenSize+inputSize,hiddenSize);
        Wf.label="Wf";
        this.Wi=init.init(hiddenSize+inputSize,hiddenSize);
        Wi.label="Wi";
        this.Wo=init.init(hiddenSize+inputSize,hiddenSize);
        Wo.label="Wo";
        this.Wc=init.init(hiddenSize+inputSize,hiddenSize);
        Wc.label="Wc";

        this.hiddenSize=hiddenSize;
        this.inputSize=inputSize;
        hiddenState=new MemoryState(null);
        cellState=new MemoryState(null);

        if (this.useBias==true) {
            this.Bi = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_i").zeros();
            this.Bf = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_f").zeros();
            this.Bo = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_o").zeros();
            this.Bc = new Tensor(1, hiddenSize, new HashSet<>(), "Bias_c").zeros();
        }

    }



    @Override
    public Tensor[] forward(Tensor [] input){
        int[] packedSizes =null;
        out =new Tensor[input.length];
        if (mask!=null) {
            packedSizes = getPackedBatchesSizes();
        }

        if (hiddenState.get()==null) {
            hiddenState= new MemoryState(new Tensor(input[0].rows,hiddenSize,new HashSet<>(),"hidden state"));
        }
        if (cellState.get()==null){
            cellState=new MemoryState(new Tensor(input[0].rows,hiddenSize,new HashSet<>(),"cell state"));
        }




        //let's calculate for each time step
        for (int step=0;step<input.length;step++){
            input[step].label="data("+step+")";
            if (mask!=null) {
                int[] indexes= IntStream.rangeClosed(0, packedSizes[step]-1).toArray();
                out[step] = lstmCell(input[step].mapVec(indexes), hiddenState.get().mapVec(indexes),cellState.get().mapVec(indexes));
            }
            else {
                out[step] = lstmCell(input[step], hiddenState.get(),cellState.get());
            }
            hiddenState.get().label="h("+step+")";
            cellState.get().label="c("+step+")";
        }

        return out;
    }

    public Tensor lstmCell(Tensor input, Tensor hiddenState, Tensor cellState){
        Tensor forgetGate=null;
        Tensor inputGate=null;
        Tensor outputGate=null;
        Tensor candidateCellState=null;
        Tensor z=input.concatRight(hiddenState);

        if (useBias){
            forgetGate=z.mul(Wf).addb(Bf).sigmoid();
            inputGate=z.mul(Wi).addb(Bi).sigmoid();
            outputGate=z.mul(Wo).addb(Bo).sigmoid();
            candidateCellState=z.mul(Wc).addb(Bc).tanh();

        }

        else {
            forgetGate=z.mul(Wf).sigmoid();
            inputGate=z.mul(Wi).sigmoid();
            outputGate=z.mul(Wo).sigmoid();
            candidateCellState=z.mul(Wc).tanh();
        }

        Tensor cellS=forgetGate.hadamard(cellState).add(inputGate.hadamard(candidateCellState));
        Tensor hiddenS=outputGate.hadamard(cellS.tanh());

        this.cellState=new MemoryState(cellS);
        this.hiddenState=new MemoryState(hiddenS);

        return hiddenS;
    }


    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(this.Wf);
        params.add(this.Wi);
        params.add(this.Wo);
        params.add(this.Wc);
        if (useBias){
            params.add(this.Bf);
            params.add(this.Bi);
            params.add(this.Bo);
            params.add(this.Bc);
        }

        return params;
    }

    @Override
    public HashSet<MemoryState> memoryList() {
        HashSet <MemoryState> memoryState =new HashSet<>();
        memoryState.add(this.hiddenState);
        memoryState.add(this.cellState);
        return memoryState;
    }

    @Override
    public void initParameters(Initializer init) {

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
