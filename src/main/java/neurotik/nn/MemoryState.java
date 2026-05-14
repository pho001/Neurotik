package neurotik.nn;

import neurotik.tensor.Tensor;

import java.util.HashSet;

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
        state=new Tensor(rows,cols,new HashSet<>(),label);
    }

    public Tensor get(){
        return state;
    }




}
