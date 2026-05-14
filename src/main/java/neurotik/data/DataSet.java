package neurotik.data;

import neurotik.encoding.Encoder;
import neurotik.tensor.Tensor;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.List;

public abstract class DataSet<T> {

    List<T> data;

    int seqLength;

    public double devRatio=0.1;
    public double testRatio=0.1;
    public double trainRatio=0.8;


    public DataSet(List<T> data){
        this.data=data;
    }



    public DataSet<T> getBatch(int batchSize, int startIndex){
        return getLines(startIndex,startIndex+batchSize);
    };



    public DataSet<T> trainingSet(){
        int bottomIndex = 0;
        int upperIndex=(int)Math.round(this.data.size()*trainRatio);
        return getLines(bottomIndex,upperIndex);
    }

    public DataSet<T> testSetData(){
        int upperIndex=(int)Math.round(this.data.size()*(trainRatio+testRatio));
        int bottomIndex=(int)Math.round(this.data.size()*(trainRatio));
        return getLines(bottomIndex,upperIndex);
    }

    public DataSet<T> devSetData(){
        int bottomIndex=(int)Math.round(this.data.size()*(trainRatio+testRatio));
        int upperIndex=this.data.size();
        return getLines(bottomIndex,upperIndex);

    }

    public DataSet<T> getLines(int bottomIndex,int upperIndex){
        if (bottomIndex < 0 || upperIndex > data.size() || bottomIndex > upperIndex) {
            throw new IndexOutOfBoundsException("Invalid indices");
        }
        List<T> inputs=new ArrayList<>();
        List<T> targets=new ArrayList<>();
        for (int i=bottomIndex;i<upperIndex;i++){
            inputs.add(this.data.get(i));
            targets.add(this.data.get(i));
        }
        return createInstance(inputs);
    }

    public abstract DataSet<T> createInstance(List<T> inputs);

    public List<T> getInputs(){
        return this.data;
    }



    public int getSetSize(){
        return data.size();
    }

    public List<T> getData(){
        return this.data;
    }

    public abstract DataSet<T> getSubSeq(int startIndex,int endIndex);
    public abstract DataSet<T> setSequences(int length,boolean usePadding);

    public abstract int[] getMask();

    public abstract Tensor[] encode(Encoder enc);

    public abstract StringDataSet getSubSeq(int endOffset);

    public DataSet<T> padSortedSequences(String paddingString){
        throw new UnsupportedOperationException("Operation not supported.");
    }
    public int[] getPackedBatchesSizes(int[] mask){
        throw new UnsupportedOperationException("Operation not supported.");
    }



}
