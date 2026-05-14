package neurotik.nn.layers;

import neurotik.nn.Layer;
import tensor.DataType;
import tensor.Tensor;

import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class EmbeddingLayer extends Layer {
    private final int vocabularySize;
    private final int embeddingSize;
    private Tensor embeddings;

    public EmbeddingLayer(int vocabularySize, int embeddingSize) {
        if (vocabularySize <= 0 || embeddingSize <= 0) {
            throw new IllegalArgumentException("Embedding dimensions must be positive.");
        }
        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;
        this.embeddings = randomEmbeddings(vocabularySize, embeddingSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        return embeddings.gatherAxis(input, 0);
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet<Tensor> params = new HashSet<>();
        params.add(embeddings);
        return params;
    }

    @Override
    public void initParameters(neurotik.nn.init.Initializer init) {
        this.embeddings = randomEmbeddings(vocabularySize, embeddingSize);
    }

    private static Tensor randomEmbeddings(int vocabularySize, int embeddingSize) {
        Random random = new Random();
        double[] data = new double[vocabularySize * embeddingSize];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian();
        }
        return new Tensor(data, new int[]{vocabularySize, embeddingSize}, List.of(), "Embeddings", DataType.FLOAT64)
                .trainableParameter();
    }
}
