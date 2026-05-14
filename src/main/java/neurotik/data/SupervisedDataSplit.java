package neurotik.data;

public record SupervisedDataSplit<X, Y>(
        SupervisedDataset<X, Y> train,
        SupervisedDataset<X, Y> test,
        SupervisedDataset<X, Y> dev) {
}
