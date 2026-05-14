package neurotik.data;

import neurotik.encoding.Encoder;
import neurotik.tensor.Tensor;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataLoader<T> {

    DataSet<T> inputs;
    DataSet<T> targets;



    int seqLength;

    public double devRatio=0.1;
    public double testRatio=0.1;
    public double trainRatio=0.8;


    public DataLoader(DataSet<T> inputs,DataSet<T> targets){
        this.inputs=inputs;
        this.targets=targets;
    }


    public DataLoader<T> trainingSet(){
        int bottomIndex=0;

        int upperIndex=(int)Math.round(this.inputs.getSetSize()*trainRatio);
        return getLines(bottomIndex,upperIndex);
    }

    public DataLoader<T> testSetData(){
        int upperIndex=(int)Math.round(this.inputs.getSetSize()*(trainRatio+testRatio));
        int bottomIndex=(int)Math.round(this.inputs.getSetSize()*(trainRatio));
        return getLines(bottomIndex,upperIndex);
    }

    public DataLoader<T> devSetData(){
        int bottomIndex=(int)Math.round(this.inputs.getSetSize()*(trainRatio+testRatio));
        int upperIndex=this.inputs.getData().size();
        return getLines(bottomIndex,upperIndex);
    }

    public DataLoader<T> getBatch(int batchSize,int startIndex){
        return new DataLoader<T>(inputs.getBatch(batchSize,startIndex),targets.getBatch(batchSize,startIndex));
    }

    public DataLoader<T> getLines(int bottomIndex,int upperIndex){

        if (bottomIndex < 0 || upperIndex > inputs.getData().size() || bottomIndex > upperIndex) {
            throw new IndexOutOfBoundsException("Invalid indices");
        }
        inputs.getLines(bottomIndex,upperIndex);

        return new DataLoader<T>(inputs.getLines(bottomIndex,upperIndex),targets.getLines(bottomIndex,upperIndex));
    }

    public int getSetSize(){
        return inputs.getSetSize();
    }

    public int[] getMask(){
        return inputs.getMask();
    }

    public DataSet<T> getInputs(){
        return this.inputs;
    }

    public DataSet<T> getTargets(){
        return this.targets;
    }

    public Tensor[] encodeInputs(Encoder enc) {return this.targets.encode(enc);}

    public Tensor[] encodeTargets(Encoder enc) {return this.targets.encode(enc);}


    public int[] getPackedBatchesSizes(){
        int[] mask=this.getMask();
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
