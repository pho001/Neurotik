package neurotik.nn;

import tensor.Tensor;

import java.util.List;

public final class TensorTimeOps {
    private TensorTimeOps() {
    }

    public static Tensor stackTime(List<Tensor> steps) {
        if (steps.isEmpty()) {
            throw new IllegalArgumentException("At least one timestep is required.");
        }
        Tensor[] expanded = new Tensor[steps.size()];
        for (int i = 0; i < steps.size(); i++) {
            expanded[i] = steps.get(i).expandDims(0);
        }
        return Tensor.concat(0, expanded);
    }
}
