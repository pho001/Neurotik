package neurotik.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SupervisedDataset<X, Y> implements Iterable<Sample<X, Y>> {
    private final List<Sample<X, Y>> samples;

    public SupervisedDataset(List<Sample<X, Y>> samples) {
        this.samples = new ArrayList<>(samples);
    }

    public static <X, Y> SupervisedDataset<X, Y> of(List<X> inputs, List<Y> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size.");
        }
        List<Sample<X, Y>> samples = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            samples.add(new Sample<>(inputs.get(i), targets.get(i)));
        }
        return new SupervisedDataset<>(samples);
    }

    public int size() {
        return samples.size();
    }

    public Sample<X, Y> get(int index) {
        return samples.get(index);
    }

    public List<Sample<X, Y>> samples() {
        return Collections.unmodifiableList(samples);
    }

    @Override
    public Iterator<Sample<X, Y>> iterator() {
        return samples().iterator();
    }

    public SupervisedDataset<X, Y> slice(int fromInclusive, int toExclusive) {
        if (fromInclusive < 0 || toExclusive > samples.size() || fromInclusive > toExclusive) {
            throw new IndexOutOfBoundsException("Invalid supervised dataset slice.");
        }
        return new SupervisedDataset<>(samples.subList(fromInclusive, toExclusive));
    }

    public SupervisedDataset<X, Y> shuffle(Random random) {
        List<Sample<X, Y>> shuffled = new ArrayList<>(samples);
        Collections.shuffle(shuffled, random);
        return new SupervisedDataset<>(shuffled);
    }

    public SupervisedDataSplit<X, Y> split(double trainRatio, double testRatio, double devRatio) {
        double sum = trainRatio + testRatio + devRatio;
        if (trainRatio < 0.0 || testRatio < 0.0 || devRatio < 0.0 || Math.abs(sum - 1.0) > 1e-9) {
            throw new IllegalArgumentException("Split ratios must be non-negative and sum to 1.0.");
        }
        int trainEnd = (int) Math.round(samples.size() * trainRatio);
        int testEnd = (int) Math.round(samples.size() * (trainRatio + testRatio));
        return new SupervisedDataSplit<>(
                slice(0, trainEnd),
                slice(trainEnd, testEnd),
                slice(testEnd, samples.size()));
    }
}
