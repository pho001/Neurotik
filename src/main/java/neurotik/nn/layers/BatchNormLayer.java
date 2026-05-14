package neurotik.nn.layers;

import neurotik.nn.Layer;
import neurotik.nn.init.Initializer;
import tensor.DataType;
import tensor.Tensor;

import java.util.HashSet;
import java.util.List;

public class BatchNormLayer extends Layer {
    private static final double DEFAULT_EPSILON = 1e-5;
    private static final double DEFAULT_MOMENTUM = 0.9;

    private final int features;
    private final int channelAxis;
    private final double epsilon;
    private final double momentum;

    private Tensor gamma;
    private Tensor beta;
    private Tensor runningMean;
    private Tensor runningVariance;
    private Tensor out;

    public BatchNormLayer(int features) {
        this(features, 1, DEFAULT_EPSILON, DEFAULT_MOMENTUM);
    }

    public BatchNormLayer(int features, int channelAxis) {
        this(features, channelAxis, DEFAULT_EPSILON, DEFAULT_MOMENTUM);
    }

    public BatchNormLayer(int features, int channelAxis, double epsilon) {
        this(features, channelAxis, epsilon, DEFAULT_MOMENTUM);
    }

    public BatchNormLayer(int features, int channelAxis, double epsilon, double momentum) {
        if (features <= 0) {
            throw new IllegalArgumentException("BatchNormLayer features must be positive.");
        }
        if (epsilon <= 0.0) {
            throw new IllegalArgumentException("BatchNormLayer epsilon must be positive.");
        }
        if (momentum < 0.0 || momentum >= 1.0) {
            throw new IllegalArgumentException("BatchNormLayer momentum must be in range [0, 1).");
        }
        this.features = features;
        this.channelAxis = channelAxis;
        this.epsilon = epsilon;
        this.momentum = momentum;
        this.gamma = parameter(ones(features), "BatchNorm gamma");
        this.beta = parameter(new double[features], "BatchNorm beta");
        this.runningMean = state(new double[features], "BatchNorm running mean");
        this.runningVariance = state(ones(features), "BatchNorm running variance");
    }

    @Override
    public Tensor forward(Tensor input) {
        this.out = normalize(input);
        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet<Tensor> params = new HashSet<>();
        params.add(gamma);
        params.add(beta);
        return params;
    }

    @Override
    public void initParameters(Initializer init) {
        this.gamma = parameter(ones(features), "BatchNorm gamma");
        this.beta = parameter(new double[features], "BatchNorm beta");
        this.runningMean = state(new double[features], "BatchNorm running mean");
        this.runningVariance = state(ones(features), "BatchNorm running variance");
    }

    private Tensor normalize(Tensor input) {
        int axis = normalizeAxis(channelAxis, input.getShape().length);
        if (input.getDimensionAt(axis) != features) {
            throw new IllegalArgumentException("BatchNormLayer expected " + features
                    + " features at axis " + axis + " but got " + input.getDimensionAt(axis) + ".");
        }

        int[] parameterShape = channelParameterShape(input.getShape(), axis);
        Tensor mean;
        Tensor variance;
        if (learningMode) {
            mean = reduceAllButChannel(input, axis);
            variance = reduceAllButChannel(input.sub(mean).pow(2), axis);
            updateRunningStats(mean, variance);
        } else {
            mean = runningMean.reshape(parameterShape);
            variance = runningVariance.reshape(parameterShape);
        }

        Tensor normalized = input.sub(mean)
                .div(variance.add(Tensor.scalar(epsilon, input.getDataType())).sqrt());
        return normalized
                .mul(gamma.reshape(parameterShape))
                .add(beta.reshape(parameterShape));
    }

    private static Tensor parameter(double[] data, String label) {
        return new Tensor(data, new int[]{data.length}, List.of(), label, DataType.FLOAT64)
                .trainableParameter();
    }

    private static Tensor state(double[] data, String label) {
        return new Tensor(data, new int[]{data.length}, List.of(), label, DataType.FLOAT64);
    }

    private Tensor reduceAllButChannel(Tensor input, int channelAxis) {
        Tensor reduced = input;
        for (int axis = 0; axis < input.getShape().length; axis++) {
            if (axis != channelAxis) {
                reduced = reduced.mean(axis, true);
            }
        }
        return reduced;
    }

    private void updateRunningStats(Tensor batchMean, Tensor batchVariance) {
        batchMean.compute();
        batchVariance.compute();
        runningMean.setData(exponentialMovingAverage(runningMean.toDoubleArrayCopy(), batchMean.toDoubleArrayCopy()));
        runningVariance.setData(exponentialMovingAverage(runningVariance.toDoubleArrayCopy(), batchVariance.toDoubleArrayCopy()));
    }

    private double[] exponentialMovingAverage(double[] running, double[] batch) {
        double[] updated = new double[running.length];
        for (int i = 0; i < updated.length; i++) {
            updated[i] = momentum * running[i] + (1.0 - momentum) * batch[i];
        }
        return updated;
    }

    private static int normalizeAxis(int axis, int rank) {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for rank " + rank + ".");
        }
        return normalized;
    }

    private static int[] channelParameterShape(int[] inputShape, int channelAxis) {
        int[] shape = new int[inputShape.length];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = 1;
        }
        shape[channelAxis] = inputShape[channelAxis];
        return shape;
    }

    private static double[] ones(int size) {
        double[] data = new double[size];
        for (int i = 0; i < data.length; i++) {
            data[i] = 1.0;
        }
        return data;
    }
}
