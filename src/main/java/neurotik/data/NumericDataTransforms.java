package neurotik.data;

import java.util.ArrayList;
import java.util.List;

public final class NumericDataTransforms {
    private NumericDataTransforms() {
    }

    public static DataSet<double[]> windows(DataSet<double[]> seriesSet, int sequenceLength) {
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("Sequence length must be positive.");
        }
        List<double[]> windows = new ArrayList<>();
        for (double[] series : seriesSet.data()) {
            for (int start = 0; start <= series.length - sequenceLength; start++) {
                double[] window = new double[sequenceLength];
                System.arraycopy(series, start, window, 0, sequenceLength);
                windows.add(window);
            }
        }
        return new DataSet<>(windows);
    }

    public static DataSet<double[]> sliceOffsets(DataSet<double[]> dataSet, int startOffset, int endOffset) {
        List<double[]> out = new ArrayList<>();
        for (double[] values : dataSet.data()) {
            int startIndex = startOffset;
            int endIndexInclusive = values.length + endOffset - 1;
            if (startIndex < 0 || startIndex >= values.length
                    || endIndexInclusive < 0 || endIndexInclusive >= values.length
                    || startIndex > endIndexInclusive) {
                throw new IllegalArgumentException("Invalid numeric offsets.");
            }
            int newArrayLength = endIndexInclusive - startIndex + 1;
            double[] slice = new double[newArrayLength];
            System.arraycopy(values, startIndex, slice, 0, newArrayLength);
            out.add(slice);
        }
        return new DataSet<>(out);
    }
}
