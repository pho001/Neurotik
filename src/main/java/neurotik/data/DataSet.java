package neurotik.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataSet<T> {
    private final List<T> data;

    public DataSet(List<T> data) {
        this.data = new ArrayList<>(data);
    }

    public static <T> DataSet<T> of(List<T> data) {
        return new DataSet<>(data);
    }

    public int size() {
        return data.size();
    }

    public int getSetSize() {
        return size();
    }

    public List<T> data() {
        return Collections.unmodifiableList(data);
    }

    public List<T> getData() {
        return data();
    }

    public T get(int index) {
        return data.get(index);
    }

    public DataSet<T> slice(int fromInclusive, int toExclusive) {
        if (fromInclusive < 0 || toExclusive > data.size() || fromInclusive > toExclusive) {
            throw new IndexOutOfBoundsException("Invalid dataset slice.");
        }
        return new DataSet<>(data.subList(fromInclusive, toExclusive));
    }

    public DataSet<T> batch(int batchSize, int startIndex) {
        return slice(startIndex, Math.min(startIndex + batchSize, data.size()));
    }

    public DataSet<T> shuffle(Random random) {
        List<T> shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, random);
        return new DataSet<>(shuffled);
    }

    public DataSplit<T> split(double trainRatio, double testRatio, double devRatio) {
        validateRatios(trainRatio, testRatio, devRatio);
        int trainEnd = (int) Math.round(data.size() * trainRatio);
        int testEnd = (int) Math.round(data.size() * (trainRatio + testRatio));
        return new DataSplit<>(
                slice(0, trainEnd),
                slice(trainEnd, testEnd),
                slice(testEnd, data.size()));
    }

    private static void validateRatios(double trainRatio, double testRatio, double devRatio) {
        double sum = trainRatio + testRatio + devRatio;
        if (trainRatio < 0.0 || testRatio < 0.0 || devRatio < 0.0 || Math.abs(sum - 1.0) > 1e-9) {
            throw new IllegalArgumentException("Split ratios must be non-negative and sum to 1.0.");
        }
    }
}
