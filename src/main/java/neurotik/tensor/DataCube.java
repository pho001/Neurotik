package neurotik.tensor;

import neurotik.encoding.IEncoder;

import java.util.HashSet;
import java.util.List;


/*
Loads data to data cube:
x axis - encoded value
y axis - batch
z axis - context
Example - word "acme" will be loaded as
x axis - will represent encoded values of letter in given context position. For first position, it will represent letter "a". Dimension of x depends on encoding.
y axis - represent samples in batch. Our example is batch with one line, so dimension is 1 in our case.
z axis - represent context position. Dimension is equal to context length. In our case, dim = 4.
Each slice of this cube by z axis will represent encoding for each context position on z axis.
 */
public class DataCube {
    Tensor[] dataCube;

    int contextLength;
    boolean isLastCharExpectedOutput;



    public DataCube(int contextLength,boolean isLastCharExpectedOutput, List<String> data, IEncoder enc)
    {
        if (isLastCharExpectedOutput)
            dataCube=new Tensor[contextLength-1];
        else
            dataCube=new Tensor[contextLength];
        this.isLastCharExpectedOutput=isLastCharExpectedOutput;
        this.contextLength=contextLength;
        fillDataCube(data,enc);
    }

    public void fillDataCube(List<String> batch,IEncoder encoder){
        for (int c=0;c< dataCube.length;c++){
            Tensor contextLayer=null;
            Tensor encoded=null;
            char[] inputs = new char [batch.size()];
            int i=0;
            for (String s : batch) {
                inputs[i]=s.charAt(c);
                i++;
            }
            contextLayer=encoder.encode(inputs);
            dataCube[c]=contextLayer;
        }
    }

    public Tensor[] getDataCube(List<String> batch) {
        return dataCube;
    }

    public Tensor getExpectedOutput(List<String> batch,IEncoder encoder) {
        Tensor contextLayer=null;
        Tensor encoded=null;
        char[] inputs = new char [batch.size()];
        int i=0;
        for (String s : batch) {

            inputs[i]=s.charAt(contextLength);
            i++;
        }
        return encoder.encode(inputs);
    }

    public Tensor[] flatten(int factor){
        if (dataCube.length%factor!=0){
            if (dataCube.length==1){
                return dataCube;
            }
            throw new RuntimeException("Unable to reduce dimension.");
        }
        Tensor prev=null;
        Tensor [] out= new Tensor[dataCube.length/factor];
        int depth=dataCube.length;
        int k=0;
        for (int i=0;i<depth;i++){
            if(i%factor==0){
                for (int j=0;j<factor;j++){
                    if (j==0){
                        out[k]=dataCube[j+i];
                    }
                    else {
                        //this.out[k]=this.out[k].join(input[j], Tensor.Join.RIGHT);
                        out[k]=out[k].join(dataCube[j+i], Tensor.Join.RIGHT);

                    }
                }
                k++;
            }
        }
        return out;
    }
}
