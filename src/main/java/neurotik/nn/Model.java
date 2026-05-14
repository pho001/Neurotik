package neurotik.nn;

import neurotik.data.DataLoader;
import neurotik.data.SequenceBatch;
import neurotik.data.TextVocabulary;
import neurotik.nn.init.Initializer;
import neurotik.nn.optim.Optimizer;
import neurotik.tensor.MathHelper;
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.loss.LossReduction;

import java.util.Arrays;
import java.util.HashSet;
import java.util.OptionalDouble;
import java.util.Random;

public abstract class Model {
    protected boolean learningMode;

    protected int[] hiddenLayers;
    protected Layers layers;
    protected int contextLength;
    protected boolean useBias;
    protected boolean isTextBased;
    protected int featuresIn;
    protected int featuresOut;
    protected Initializer init;
    protected TextVocabulary vocabulary;

    public Model(int contextLength, int featuresIn, int[] hiddenLayers, TextVocabulary vocabulary, boolean useBias, Initializer init) {
        this.contextLength = contextLength;
        this.hiddenLayers = hiddenLayers;
        this.vocabulary = vocabulary;
        this.useBias = useBias;
        this.featuresIn = featuresIn;
        this.featuresOut = vocabulary.size();
        this.init = init;
        this.learningMode = false;
        this.isTextBased = true;
        this.layers = topology();
    }

    public Model(int contextLength, int featuresIn, int[] hiddenLayers, int featuresOut, boolean useBias, Initializer init) {
        this.contextLength = contextLength;
        this.hiddenLayers = hiddenLayers;
        this.useBias = useBias;
        this.featuresIn = featuresIn;
        this.featuresOut = featuresOut;
        this.init = init;
        this.learningMode = false;
        this.isTextBased = false;
        this.layers = topology();
    }

    public HashSet<Tensor> parameters() {
        HashSet<Tensor> params = new HashSet<>();
        for (Layer layer : layers.layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    public HashSet<MemoryState> memoryList() {
        HashSet<MemoryState> memoryStateList = new HashSet<>();
        for (Layer layer : layers.layers) {
            memoryStateList.addAll(layer.memoryList());
        }
        return memoryStateList;
    }

    public void train(DataLoader<SequenceBatch> dataLoader, int epochs, Optimizer optimizer) {
        System.out.println("Parameters count: " + this.getParamsCount());
        int lines = 100;
        this.setLearningMode(true);
        for (int epoch = 0; epoch < epochs; epoch++) {
            double durationInMillis = 0;
            double cumLoss = 0;
            for (int batchIndex = 0; batchIndex < dataLoader.size(); batchIndex++) {
                long startTime = System.nanoTime();
                this.resetMemoryStates();
                Tensor loss = getLoss(dataLoader.getBatch(batchIndex));

                for (Tensor parameter : parameters()) {
                    TensorInternalAccess.clearGradient(parameter);
                }
                loss.compute(CompileMode.TRAINING);
                updateParameters(optimizer, epoch);

                long endTime = System.nanoTime();
                durationInMillis += (endTime - startTime) / 1e6;
                cumLoss += loss.scalarAsDouble();

                if (batchIndex % lines == 0 && batchIndex > 0) {
                    System.out.println("Epoch: " + epoch
                            + " || Batch " + batchIndex + "/" + dataLoader.size()
                            + ": " + cumLoss / lines
                            + " || Avg. duration : " + durationInMillis / lines + " ms");
                    durationInMillis = 0;
                    cumLoss = 0;
                }
            }
        }
    }

    public Tensor getLoss(SequenceBatch batch) {
        Tensor predictions = forward(batch.inputs());
        if (batch.classification()) {
            return classificationLoss(predictions, batch.targets(), batch.mask());
        }
        return regressionLoss(predictions, batch.targets(), batch.mask());
    }

    protected Tensor forward(Tensor inputs) {
        Tensor out = inputs;
        for (Layer layer : layers.layers) {
            out = layer.forward(out);
        }
        return out;
    }

    public void getSetLoss(DataLoader<SequenceBatch> dataLoader) {
        double[] cumLoss = new double[dataLoader.size()];
        for (int i = 0; i < dataLoader.size(); i++) {
            this.resetMemoryStates();
            Tensor loss = getLoss(dataLoader.getBatch(i));
            loss.compute();
            cumLoss[i] = loss.scalarAsDouble();
        }
        OptionalDouble avg = Arrays.stream(cumLoss).average();
        if (avg.isPresent()) {
            System.out.println("Test set loss: " + avg.getAsDouble());
        }
    }

    public abstract void generate(int samples);

    public void generate(String prompt, int samples) {
    }

    public void generate(double[] initValues, int samples) {
    }

    public void setLearningMode(boolean learningMode) {
        this.learningMode = learningMode;
        for (Layer layer : layers.layers) {
            layer.setLearningMode(learningMode);
        }
    }

    public void updateParameters(Optimizer opt, int epoch) {
        for (Tensor p : parameters()) {
            opt.update(p, epoch + 1);
        }
    }

    public void resetMemoryStates() {
        for (MemoryState memoryState : memoryList()) {
            if (memoryState != null) {
                memoryState.reset();
            }
        }
    }

    public abstract Layers topology();

    public int getParamsCount() {
        int paramsCount = 0;
        for (Tensor p : parameters()) {
            int tensorParams = 1;
            for (int dimension : p.getShape()) {
                tensorParams *= dimension;
            }
            paramsCount += tensorParams;
        }
        return paramsCount;
    }

    public void initParameters(Initializer initializer) {
        for (Layer l : layers.layers) {
            l.initParameters(initializer);
        }
    }

    public void setMask(int[] mask) {
        for (Layer l : layers.layers) {
            l.setMask(mask);
        }
    }

    private Tensor classificationLoss(Tensor predictions, Tensor targets, Tensor mask) {
        if (predictions.getShape().length == 2) {
            Tensor target = targets.getShape().length == 2 ? targets.select(0, 0) : targets;
            Tensor loss = predictions.crossEntropyLossFromIndices(target, 1, LossReduction.NONE);
            Tensor selectedMask = mask.getShape().length == 2 ? mask.select(0, 0) : mask;
            return maskedMean(loss, selectedMask);
        }
        Tensor loss = predictions.crossEntropyLossFromIndices(targets, 2, LossReduction.NONE);
        return maskedMean(loss, mask);
    }

    private Tensor regressionLoss(Tensor predictions, Tensor targets, Tensor mask) {
        Tensor alignedTargets = targets;
        Tensor alignedMask = mask;
        if (predictions.getShape().length == 2 && targets.getShape().length == 3) {
            alignedTargets = targets.select(0, 0);
            alignedMask = mask.select(0, 0);
        }
        Tensor loss = predictions.sub(alignedTargets).pow(2);
        if (loss.getShape().length == 3 && alignedMask.getShape().length == 2) {
            alignedMask = alignedMask.expandDims(2);
        }
        return maskedMean(loss, alignedMask);
    }

    private Tensor maskedMean(Tensor values, Tensor mask) {
        Tensor numericMask = mask.cast(DataType.FLOAT64);
        return values.mul(numericMask).sum().div(numericMask.sum());
    }

    protected Tensor textInput(String value) {
        int[] data = new int[value.length()];
        for (int i = 0; i < value.length(); i++) {
            data[i] = vocabulary.indexOf(value.charAt(i));
        }
        return new Tensor(data, new int[]{value.length(), 1}, java.util.List.of(), "prompt", DataType.INT32);
    }

    protected char sampleNextChar(Tensor predictions, Random random) {
        Tensor logits = predictions;
        if (predictions.getShape().length == 3) {
            logits = predictions.select(0, predictions.getDimensionAt(0) - 1);
        }
        Tensor probs = logits.softmax(1).compute();
        double[] probabilities = new double[probs.getDimensionAt(1)];
        System.arraycopy(probs.toDoubleArrayCopy(), 0, probabilities, 0, probabilities.length);
        int index = MathHelper.sampleFromMultinomial(1, probabilities, random)[0];
        return vocabulary.charAt(index);
    }
}
