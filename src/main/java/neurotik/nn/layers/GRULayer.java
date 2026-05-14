package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.nn.MemoryState;
import neurotik.nn.TensorTimeOps;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

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



    Tensor out;



    public GRULayer(int inputSize, int hiddenSize,boolean useBias,Initializer init){
        this.useBias=useBias;
        this.step=0;

        if (this.useBias==true) {
            this.bias_h = new Tensor(new double[hiddenSize], new int[]{1, hiddenSize}, List.of(), "Bias_h", DataType.FLOAT64)
                    .trainableParameter();
        }

        this.Wu=init.init(inputSize+hiddenSize,hiddenSize);
        this.Wr=init.init(inputSize+hiddenSize,hiddenSize);
        this.Wh=init.init(inputSize+hiddenSize,hiddenSize);

        this.hiddenSize=hiddenSize;
        this.inputSize=inputSize;


        lastMemoryState=new MemoryState();


    }



    @Override
    public Tensor forward(Tensor input){
        int time = input.getDimensionAt(0);
        int batch = input.getDimensionAt(1);
        List<Tensor> steps = new ArrayList<>();

        //if hidden state wasn't initialized yet
        if (lastMemoryState.get()==null){
            lastMemoryState=new MemoryState(new Tensor(new double[batch * hiddenSize], new int[]{batch, hiddenSize}, List.of(), "h(-1)", DataType.FLOAT64));
        }
        //let's calculate for each time step
        for (int step=0;step<time;step++){
            Tensor stepInput = input.select(0, step);
            stepInput.setLabel("data("+step+")");
            steps.add(gruCell(stepInput, lastMemoryState.get()));
            lastMemoryState.get().setLabel("h("+step+")");
        }

        out = TensorTimeOps.stackTime(steps);
        return out;
    }

    public Tensor gruCell(Tensor input,Tensor hprev){
        Tensor updateGate=null;
        Tensor resetGate=null;
        Tensor candidateHidState=null;



        Tensor z= Tensor.concat(1, input, hprev);

        if (useBias) {
            updateGate=z.matmul(Wu).add(bias_h).sigmoid();
            resetGate=z.matmul(Wr).add(bias_h).sigmoid();
            candidateHidState=Tensor.concat(1, input, resetGate.mul(hprev)).matmul(Wh).add(bias_h).tanh();
        }
        else {
            updateGate=z.matmul(Wu).sigmoid();
            resetGate=z.matmul(Wr).sigmoid();
            candidateHidState=Tensor.concat(1, input, resetGate.mul(hprev)).matmul(Wh).tanh();
        }


        Tensor ones=Tensor.onesLike(updateGate);
        Tensor hidden=ones.sub(updateGate).mul(hprev).add(updateGate.mul(candidateHidState));
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
        Wh.setLabel("Wh");
        Wu=init.init(inputSize+hiddenSize,hiddenSize);
        Wu.setLabel("Wu");
        Wr=init.init(inputSize+hiddenSize,hiddenSize);
        Wr.setLabel("Wr");
    }
}
