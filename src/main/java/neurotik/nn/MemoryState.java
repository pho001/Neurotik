package neurotik.nn;

import tensor.Tensor;
import tensor.DataType;

import java.util.List;

public class MemoryState {
    Tensor state;

    public MemoryState(){
       state= null;
    }

    public MemoryState(Tensor state){
        this.state=state;
    }

    public void reset(){
        state=null;
    }

    public void init(int rows, int cols,String label){
        state= new Tensor(new double[rows * cols], new int[]{rows, cols}, List.of(), label, DataType.FLOAT64);
    }

    public Tensor get(){
        return state;
    }




}
