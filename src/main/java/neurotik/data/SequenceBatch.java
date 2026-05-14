package neurotik.data;

import tensor.Tensor;

public record SequenceBatch(
        Tensor inputs,
        Tensor targets,
        Tensor mask,
        int[] lengths,
        boolean classification) implements Batch {
}
