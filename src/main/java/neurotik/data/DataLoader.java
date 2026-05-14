package neurotik.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class DataLoader<B extends Batch> implements Iterable<B> {
    private final List<B> batches;

    public DataLoader(List<B> batches) {
        this.batches = new ArrayList<>(batches);
    }

    public static <X, Y> Builder<X, Y> from(SupervisedDataset<X, Y> dataset) {
        return new Builder<>(dataset);
    }

    public int size() {
        return batches.size();
    }

    public B getBatch(int index) {
        return batches.get(index);
    }

    public List<B> batches() {
        return Collections.unmodifiableList(batches);
    }

    @Override
    public Iterator<B> iterator() {
        return batches().iterator();
    }

    public DataLoader<B> slice(int fromInclusive, int toExclusive) {
        if (fromInclusive < 0 || toExclusive > batches.size() || fromInclusive > toExclusive) {
            throw new IndexOutOfBoundsException("Invalid dataloader slice.");
        }
        return new DataLoader<>(batches.subList(fromInclusive, toExclusive));
    }

    public static final class Builder<X, Y> {
        private final SupervisedDataset<X, Y> dataset;
        private int batchSize = 32;
        private boolean shuffle = false;
        private Random random = new Random();

        private Builder(SupervisedDataset<X, Y> dataset) {
            this.dataset = dataset;
        }

        public Builder<X, Y> batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("Batch size must be positive.");
            }
            this.batchSize = batchSize;
            return this;
        }

        public Builder<X, Y> shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }

        public Builder<X, Y> random(Random random) {
            this.random = random;
            return this;
        }

        public <B extends Batch> DataLoader<B> collator(Collator<Sample<X, Y>, B> collator) {
            SupervisedDataset<X, Y> source = shuffle ? dataset.shuffle(random) : dataset;
            List<B> batches = new ArrayList<>();
            for (int start = 0; start < source.size(); start += batchSize) {
                int end = Math.min(start + batchSize, source.size());
                batches.add(collator.collate(source.slice(start, end).samples()));
            }
            return new DataLoader<>(batches);
        }
    }
}
