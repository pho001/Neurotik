package neurotik;

import neurotik.data.DataLoader;
import neurotik.data.DataSet;
import neurotik.data.NumericDatasets;
import neurotik.data.NumericSequenceCollator;
import neurotik.data.SequenceBatch;
import neurotik.data.SupervisedDataset;
import neurotik.data.TextDatasets;
import neurotik.data.TextSequenceCollator;
import neurotik.data.TextVocabulary;
import neurotik.nn.Model;
import neurotik.nn.ModelFactory;
import neurotik.nn.init.InitializerFactory;
import neurotik.nn.layers.EmbeddingLayer;
import org.junit.jupiter.api.Test;
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DataPipelineSmokeTest {
    @Test
    void textSequenceDatasetUsesShiftedStartAndEndTokenTargets() {
        SupervisedDataset<String, String> dataset = TextDatasets.sequenceNextChar(new DataSet<>(List.of("ab")), 4, ".");

        assertEquals(1, dataset.size());
        assertEquals(".ab", dataset.get(0).input());
        assertEquals("ab.", dataset.get(0).target());
    }

    @Test
    void textCollatorBuildsTimeMajorIndexBatchAndEmbedding() {
        DataSet<String> source = new DataSet<>(List.of("ab cd"));
        TextVocabulary vocabulary = TextVocabulary.from(source, ".");

        DataLoader<SequenceBatch> loader = DataLoader
                .from(TextDatasets.fixedNextChar(source, 4, "."))
                .batchSize(2)
                .collator(TextSequenceCollator.indices(vocabulary, "."));

        SequenceBatch batch = loader.getBatch(0);
        assertArrayEquals(new int[]{3, 2}, batch.inputs().getShape());
        assertArrayEquals(new int[]{1, 2}, batch.targets().getShape());
        assertArrayEquals(new int[]{1, 2}, batch.mask().getShape());
        assertEquals(DataType.INT32, batch.inputs().getDataType());
        assertEquals(DataType.INT32, batch.targets().getDataType());
        assertEquals(DataType.BOOL, batch.mask().getDataType());
        assertTrue(batch.classification());

        Tensor embedded = new EmbeddingLayer(vocabulary.size(), 5)
                .forward(batch.inputs())
                .compute();
        assertArrayEquals(new int[]{3, 2, 5}, embedded.getShape());
    }

    @Test
    void numericCollatorBuildsTimeMajorRegressionBatch() {
        DataLoader<SequenceBatch> loader = DataLoader
                .from(NumericDatasets.sequencePrediction(new DataSet<>(List.of(new double[]{1.0, 2.0, 3.0, 4.0, 5.0})), 4))
                .batchSize(2)
                .collator(new NumericSequenceCollator());

        SequenceBatch batch = loader.getBatch(0);
        assertArrayEquals(new int[]{3, 2, 1}, batch.inputs().getShape());
        assertArrayEquals(new int[]{3, 2, 1}, batch.targets().getShape());
        assertArrayEquals(new int[]{3, 2}, batch.mask().getShape());
        assertEquals(DataType.FLOAT64, batch.inputs().getDataType());
        assertEquals(DataType.FLOAT64, batch.targets().getDataType());
        assertEquals(DataType.BOOL, batch.mask().getDataType());
    }

    @Test
    void modelLossComputesThroughTextAndNumericBatches() {
        DataSet<String> textSource = new DataSet<>(List.of("ab cd"));
        TextVocabulary vocabulary = TextVocabulary.from(textSource, ".");
        SequenceBatch textBatch = DataLoader
                .from(TextDatasets.fixedNextChar(textSource, 4, "."))
                .batchSize(2)
                .collator(TextSequenceCollator.indices(vocabulary, "."))
                .getBatch(0);
        Model textModel = ModelFactory.MLP(4, 3, new int[]{6}, vocabulary, false, InitializerFactory.kaimingInit());
        Tensor textLoss = textModel.getLoss(textBatch).compute(CompileMode.TRAINING);
        assertTrue(Double.isFinite(textLoss.scalarAsDouble()));

        SequenceBatch numericBatch = DataLoader
                .from(NumericDatasets.fixedPrediction(new DataSet<>(List.of(new double[]{1.0, 2.0, 3.0, 4.0, 5.0})), 4))
                .batchSize(2)
                .collator(new NumericSequenceCollator())
                .getBatch(0);
        Model numericModel = ModelFactory.MLP(4, 1, new int[]{4}, 1, false, InitializerFactory.kaimingInit());
        Tensor numericLoss = numericModel.getLoss(numericBatch).compute(CompileMode.TRAINING);
        assertTrue(Double.isFinite(numericLoss.scalarAsDouble()));
    }

    @Test
    void recurrentModelLossComputesThroughTimeMajorSequenceBatches() {
        DataSet<String> textSource = new DataSet<>(List.of("a"));
        TextVocabulary vocabulary = TextVocabulary.from(textSource, ".");
        SequenceBatch textBatch = DataLoader
                .from(TextDatasets.sequenceNextChar(textSource, 3, "."))
                .batchSize(1)
                .collator(TextSequenceCollator.indices(vocabulary, "."))
                .getBatch(0);

        Model rnn = ModelFactory.RNN(3, 2, new int[]{2}, vocabulary, false, InitializerFactory.kaimingInit());
        assertTrue(Double.isFinite(rnn.getLoss(textBatch).compute(CompileMode.TRAINING).scalarAsDouble()));

        Model gru = ModelFactory.GRU(3, 2, new int[]{2}, vocabulary, false, InitializerFactory.kaimingInit());
        assertTrue(Double.isFinite(gru.getLoss(textBatch).compute(CompileMode.TRAINING).scalarAsDouble()));

        Model lstm = ModelFactory.LSTM(3, 2, new int[]{2}, vocabulary, false, InitializerFactory.kaimingInit());
        assertTrue(Double.isFinite(lstm.getLoss(textBatch).compute(CompileMode.TRAINING).scalarAsDouble()));
    }
}
