package neurotik.nn.layers;

import neurotik.nn.init.Initializer;
import neurotik.nn.Layer;
import neurotik.tensor.Tensor;

import java.util.HashSet;

public class FlattenLayer extends Layer{

    int factor;
    Tensor [] out= null;

    public FlattenLayer(int factor){
        this.factor=factor;
    }




    public Tensor[] forward(Tensor [] input){
        if (input.length%factor!=0){
            if (input.length==1){
                return input;
            }
            throw new RuntimeException("Unable to reduce dimension.");
        }
        Tensor prev=null;
        this.out= new Tensor[input.length/this.factor];
        int depth=input.length;
        int k=0;
        for (int i=0;i<depth;i++){
            if(i%this.factor==0){
                for (int j=0;j<this.factor;j++){
                    if (j==0){
                        this.out[k]=input[j+i];
                    }
                    else {
                        this.out[k]=this.out[k].concatRight(input[j+i]);
                    }
                }
                k++;
            }
        }
        return this.out;
    }



    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }

    @Override
    public void initParameters(Initializer init) {

    }
}
