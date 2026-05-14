package neurotik.nn;

import neurotik.nn.init.Initializer;
import tensor.Tensor;

import java.util.*;

public abstract class Layer {
    protected boolean learningMode;
    protected boolean readyToReset=false;

    protected int[] mask;


    protected MemoryState memoryState;
    public Layer(){

        //initial mode is learning
        learningMode=true;

    }

    public abstract Tensor forward(Tensor input);



    public abstract HashSet<Tensor> parameters();



    public void setLearningMode(boolean learningMode) {
        this.learningMode=learningMode;
    }


    public HashSet<MemoryState> memoryList(){
        HashSet<MemoryState> memoryState =new HashSet<>();
        return memoryState;
    }

    public void setMask(int[] mask){
        this.mask=mask;
    }


    public void setResetState(boolean readyToReset){
        this.readyToReset=readyToReset;
    }

    public abstract void initParameters(Initializer init);

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
