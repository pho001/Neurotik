package neurotik.data;

public record DataSplit<T>(DataSet<T> train, DataSet<T> test, DataSet<T> dev) {
}
