package neurotik.data;

import neurotik.encoding.Encoder;
import neurotik.tensor.Tensor;

import java.util.ArrayList;

import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;

public class NumDataSet extends DataSet<double[]>{





    public NumDataSet(List<double[]> dataSet){
        super(dataSet);
    }

    @Override
    public NumDataSet createInstance(List<double[]> data) {
        return new NumDataSet(data);
    }

    @Override
    public NumDataSet getSubSeq(int startOffset, int endOffset) {
        int startIndex=startOffset;
        int endIndex=this.data.get(0).length+endOffset-1;
        if (startIndex < 0 || startIndex >= this.data.get(0).length || endIndex < 0 || endIndex >= this.data.get(0).length || startIndex > endIndex) {
            throw new IllegalArgumentException("Invalid offsets provided.");
        }
        int newArrayLength = endIndex - startIndex + 1;
        List<double[]> out= new ArrayList<>();
        for (double[] d : this.data) {
            double[] subArray = new double[newArrayLength];
            System.arraycopy(d, startIndex, subArray, 0, newArrayLength);
            out.add(subArray);
        }
        return createInstance(out);
    }



    @Override
    public NumDataSet setSequences(int seqLength,boolean usePadding) {
            this.seqLength=seqLength;

            List<double[]> sequences= IntStream.rangeClosed(0, data.get(0).length - seqLength)
                    .mapToObj(i -> {
                        double[] window = new double[seqLength];
                        System.arraycopy(data.get(0), i, window, 0, seqLength);
                        return window;
                    })
                    .toList();

            return createInstance(sequences);
    }
    @Override
    public StringDataSet getSubSeq(int endOffset) {
        return null;
    }



    public Tensor[] encode(Encoder enc){

            int listSize=getSetSize();
            int seqSize=this.data.get(0).length;
            Tensor [] out=new Tensor [seqSize];

            for (int i=0;i<seqSize;i++){
                double[][] slice= new double [listSize] [1];
                int j=0;
                for (double[] darray:this.data){

                    slice[j][0]=darray[i];
                    j++;
                }
                out[i]=new Tensor(slice,new HashSet<>(),"data");
            }

            return out;
    }



    @Override
    public int[] getMask() {
        return null;
    }


}
