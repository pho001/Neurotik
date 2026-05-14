package neurotik.data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class DataLoader<T> {
    private final DataSet<T> inputs;
    private final DataSet<T> targets;

    public double devRatio = 0.1;
    public double testRatio = 0.1;
    public double trainRatio = 0.8;

    public DataLoader(DataSet<T> inputs, DataSet<T> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size.");
        }
        this.inputs = inputs;
        this.targets = targets;
    }

    public static <T> DataLoader<T> supervised(DataSet<T> inputs, DataSet<T> targets) {
        return new DataLoader<>(inputs, targets);
    }

    public DataLoader<T> trainingSet() {
        return getLines(0, (int) Math.round(inputs.size() * trainRatio));
    }

    public DataLoader<T> testSetData() {
        int bottomIndex = (int) Math.round(inputs.size() * trainRatio);
        int upperIndex = (int) Math.round(inputs.size() * (trainRatio + testRatio));
        return getLines(bottomIndex, upperIndex);
    }

    public DataLoader<T> devSetData() {
        int bottomIndex = (int) Math.round(inputs.size() * (trainRatio + testRatio));
        return getLines(bottomIndex, inputs.size());
    }

    public DataLoader<T> getBatch(int batchSize, int startIndex) {
        return getLines(startIndex, Math.min(startIndex + batchSize, inputs.size()));
    }

    public DataLoader<T> getLines(int bottomIndex, int upperIndex) {
        return new DataLoader<>(inputs.slice(bottomIndex, upperIndex), targets.slice(bottomIndex, upperIndex));
    }

    public DataLoader<T> sortByInputs(Comparator<T> comparator) {
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            indexes.add(i);
        }
        indexes.sort((left, right) -> comparator.compare(inputs.get(left), inputs.get(right)));

        List<T> sortedInputs = new ArrayList<>();
        List<T> sortedTargets = new ArrayList<>();
        for (int index : indexes) {
            sortedInputs.add(inputs.get(index));
            sortedTargets.add(targets.get(index));
        }
        return new DataLoader<>(new DataSet<>(sortedInputs), new DataSet<>(sortedTargets));
    }

    public int getSetSize() {
        return inputs.size();
    }

    public DataSet<T> getInputs() {
        return inputs;
    }

    public DataSet<T> getTargets() {
        return targets;
    }

    public int[] getMask() {
        if (inputs.size() == 0 || !(inputs.get(0) instanceof String)) {
            return null;
        }
        return TextDataTransforms.lengths(strings(inputs));
    }

    public int[] getPackedBatchesSizes() {
        int[] mask = getMask();
        if (mask == null) {
            return new int[0];
        }
        return TextDataTransforms.packedBatchSizes(mask);
    }

    @SuppressWarnings("unchecked")
    private static DataSet<String> strings(DataSet<?> dataSet) {
        return (DataSet<String>) dataSet;
    }
}
