package neurotik.data;

import java.util.List;

public interface DataSetTransformer <T>{

    public DataSetTransformer<T> getSubSeq(int startIndex,int endIndex);
    public DataSetTransformer<T> setSequences(int length);

    public DataSetTransformer<T> createInstance(List<T> inputs);

    List<T> gatData();

}
