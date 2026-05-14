package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.MemoryState;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

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




    Tensor hidden;
    Tensor out;

    public LstmLayer(int inputSize, int hiddenSize,boolean useBias,Initializer init){
        this.useBias=useBias;
        this.step=0;

        this.Wf=init.init(hiddenSize+inputSize,hiddenSize);
        Wf.setLabel("Wf");
        this.Wi=init.init(hiddenSize+inputSize,hiddenSize);
        Wi.setLabel("Wi");
        this.Wo=init.init(hiddenSize+inputSize,hiddenSize);
        Wo.setLabel("Wo");
        this.Wc=init.init(hiddenSize+inputSize,hiddenSize);
        Wc.setLabel("Wc");

        this.hiddenSize=hiddenSize;
        this.inputSize=inputSize;
        hiddenState=new MemoryState(null);
        cellState=new MemoryState(null);

        if (this.useBias==true) {
            this.Bi = new Tensor(new double[hiddenSize], new int[]{1, hiddenSize}, List.of(), "Bias_i", DataType.FLOAT64).trainableParameter();
            this.Bf = new Tensor(new double[hiddenSize], new int[]{1, hiddenSize}, List.of(), "Bias_f", DataType.FLOAT64).trainableParameter();
            this.Bo = new Tensor(new double[hiddenSize], new int[]{1, hiddenSize}, List.of(), "Bias_o", DataType.FLOAT64).trainableParameter();
            this.Bc = new Tensor(new double[hiddenSize], new int[]{1, hiddenSize}, List.of(), "Bias_c", DataType.FLOAT64).trainableParameter();
        }

    }



    @Override
    public Tensor forward(Tensor input){
        int time = input.getDimensionAt(0);
        int batch = input.getDimensionAt(1);
        List<Tensor> steps = new ArrayList<>();

        if (hiddenState.get()==null) {
            hiddenState= new MemoryState(new Tensor(new double[batch * hiddenSize], new int[]{batch, hiddenSize}, List.of(), "hidden state", DataType.FLOAT64));
        }
        if (cellState.get()==null){
            cellState=new MemoryState(new Tensor(new double[batch * hiddenSize], new int[]{batch, hiddenSize}, List.of(), "cell state", DataType.FLOAT64));
        }




        //let's calculate for each time step
        for (int step=0;step<time;step++){
            Tensor stepInput = input.select(0, step);
            stepInput.setLabel("data("+step+")");
            steps.add(lstmCell(stepInput, hiddenState.get(),cellState.get()));
            hiddenState.get().setLabel("h("+step+")");
            cellState.get().setLabel("c("+step+")");
        }

        out = Tensor.stack(0, steps.toArray(new Tensor[0]));
        return out;
    }

    public Tensor lstmCell(Tensor input, Tensor hiddenState, Tensor cellState){
        Tensor forgetGate=null;
        Tensor inputGate=null;
        Tensor outputGate=null;
        Tensor candidateCellState=null;
        Tensor z=Tensor.concat(1, input, hiddenState);

        if (useBias){
            forgetGate=z.matmul(Wf).add(Bf).sigmoid();
            inputGate=z.matmul(Wi).add(Bi).sigmoid();
            outputGate=z.matmul(Wo).add(Bo).sigmoid();
            candidateCellState=z.matmul(Wc).add(Bc).tanh();

        }

        else {
            forgetGate=z.matmul(Wf).sigmoid();
            inputGate=z.matmul(Wi).sigmoid();
            outputGate=z.matmul(Wo).sigmoid();
            candidateCellState=z.matmul(Wc).tanh();
        }

        Tensor cellS=forgetGate.mul(cellState).add(inputGate.mul(candidateCellState));
        Tensor hiddenS=outputGate.mul(cellS.tanh());

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
}
