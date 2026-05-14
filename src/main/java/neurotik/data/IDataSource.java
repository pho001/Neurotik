package neurotik.data;

import java.util.List;

public interface IDataSource<T> {
    List<T> getData();
    IDataSource<T> getSubSeq(int startIndex, int endIndex);
    IDataSource<T> setSequences(int length);
    List<T> getBatch(int batchSize, int batchIndex);
    double[] getMask();
}
