package neurotik.data;

import java.util.ArrayList;
import java.util.List;

public final class NumericDatasets {
    private NumericDatasets() {
    }

    public static SupervisedDataset<double[], double[]> sequencePrediction(DataSet<double[]> seriesSet, int contextLength) {
        validateContextLength(contextLength);
        List<Sample<double[], double[]>> samples = new ArrayList<>();
        for (double[] series : seriesSet.data()) {
            for (int start = 0; start <= series.length - contextLength; start++) {
                double[] input = new double[contextLength - 1];
                double[] target = new double[contextLength - 1];
                System.arraycopy(series, start, input, 0, contextLength - 1);
                System.arraycopy(series, start + 1, target, 0, contextLength - 1);
                samples.add(new Sample<>(input, target));
            }
        }
        return new SupervisedDataset<>(samples);
    }

    public static SupervisedDataset<double[], double[]> fixedPrediction(DataSet<double[]> seriesSet, int contextLength) {
        validateContextLength(contextLength);
        List<Sample<double[], double[]>> samples = new ArrayList<>();
        for (double[] series : seriesSet.data()) {
            for (int start = 0; start <= series.length - contextLength; start++) {
                double[] input = new double[contextLength - 1];
                double[] target = new double[]{series[start + contextLength - 1]};
                System.arraycopy(series, start, input, 0, contextLength - 1);
                samples.add(new Sample<>(input, target));
            }
        }
        return new SupervisedDataset<>(samples);
    }

    private static void validateContextLength(int contextLength) {
        if (contextLength < 2) {
            throw new IllegalArgumentException("Context length must be at least 2.");
        }
    }
}
