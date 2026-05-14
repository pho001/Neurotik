package neurotik.data;

import java.util.List;

public interface IDataSet <T>{

    double DEV_RATIO=0.1;

    double TEST_RATIO=0.1;

    double TRAIN_RATIO=0.8;
    public T getSubSeq(int startIndex,int endIndex);
    public T setSequences(int context);



}
