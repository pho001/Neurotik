package neurotik.data;

import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public class NumericSequenceCollator implements Collator<Sample<double[], double[]>, SequenceBatch> {
    @Override
    public SequenceBatch collate(List<Sample<double[], double[]>> samples) {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("Cannot collate an empty numeric batch.");
        }
        int batch = samples.size();
        int inputTime = samples.get(0).input().length;
        int targetTime = samples.get(0).target().length;
        double[] inputData = new double[inputTime * batch];
        double[] targetData = new double[targetTime * batch];
        byte[] maskData = new byte[targetTime * batch];
        int[] lengths = new int[batch];

        for (int b = 0; b < batch; b++) {
            Sample<double[], double[]> sample = samples.get(b);
            if (sample.input().length != inputTime || sample.target().length != targetTime) {
                throw new IllegalArgumentException("Numeric batch samples must have uniform input and target lengths.");
            }
            lengths[b] = sample.input().length;
            for (int t = 0; t < inputTime; t++) {
                inputData[t * batch + b] = sample.input()[t];
            }
            for (int t = 0; t < targetTime; t++) {
                targetData[t * batch + b] = sample.target()[t];
                maskData[t * batch + b] = 1;
            }
        }

        return new SequenceBatch(
                new Tensor(inputData, new int[]{inputTime, batch, 1}, List.of(), "numeric inputs", DataType.FLOAT64),
                new Tensor(targetData, new int[]{targetTime, batch, 1}, List.of(), "numeric targets", DataType.FLOAT64),
                new Tensor(maskData, new int[]{targetTime, batch}, List.of(), "numeric mask", DataType.BOOL),
                lengths,
                false);
    }
}
