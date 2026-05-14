package neurotik.encoding;

import neurotik.data.DataSet;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class TensorBatchEncoder {
    private TensorBatchEncoder() {
    }

    public static Tensor[] encodeText(DataSet<String> dataSet, Encoder encoder) {
        if (dataSet.size() == 0) {
            return new Tensor[0];
        }
        int length = dataSet.get(0).length();
        Tensor[] out = new Tensor[length];
        for (int position = 0; position < length; position++) {
            char[] chars = new char[dataSet.size()];
            for (int row = 0; row < dataSet.size(); row++) {
                chars[row] = dataSet.get(row).charAt(position);
            }
            out[position] = encoder.encode(chars);
        }
        return out;
    }

    public static Tensor[] encodeNumeric(DataSet<double[]> dataSet) {
        if (dataSet.size() == 0) {
            return new Tensor[0];
        }
        int sequenceLength = dataSet.get(0).length;
        Tensor[] out = new Tensor[sequenceLength];
        for (int position = 0; position < sequenceLength; position++) {
            double[][] slice = new double[dataSet.size()][1];
            for (int row = 0; row < dataSet.size(); row++) {
                slice[row][0] = dataSet.get(row)[position];
            }
            out[position] = new Tensor(slice, List.of(), "data", DataType.FLOAT64);
        }
        return out;
    }
}
