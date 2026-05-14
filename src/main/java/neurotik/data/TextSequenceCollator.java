package neurotik.data;

import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class TextSequenceCollator implements Collator<Sample<String, String>, SequenceBatch> {
    private final TextVocabulary vocabulary;
    private final char paddingToken;

    public TextSequenceCollator(TextVocabulary vocabulary, String paddingToken) {
        if (paddingToken.length() != 1) {
            throw new IllegalArgumentException("Text padding token must be one character.");
        }
        this.vocabulary = vocabulary;
        this.paddingToken = paddingToken.charAt(0);
    }

    public static TextSequenceCollator indices(TextVocabulary vocabulary, String paddingToken) {
        return new TextSequenceCollator(vocabulary, paddingToken);
    }

    @Override
    public SequenceBatch collate(List<Sample<String, String>> samples) {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("Cannot collate an empty text batch.");
        }
        List<Sample<String, String>> sorted = new ArrayList<>(samples);
        sorted.sort(Comparator.comparingInt((Sample<String, String> sample) -> sample.input().length()).reversed());

        int batch = sorted.size();
        int inputTime = maxInputLength(sorted);
        int targetTime = maxTargetLength(sorted);
        int[] lengths = new int[batch];
        int[] inputData = new int[inputTime * batch];
        int[] targetData = new int[targetTime * batch];
        byte[] maskData = new byte[targetTime * batch];
        int paddingIndex = vocabulary.indexOf(paddingToken);

        for (int b = 0; b < batch; b++) {
            Sample<String, String> sample = sorted.get(b);
            lengths[b] = sample.input().length();
            for (int t = 0; t < inputTime; t++) {
                char value = t < sample.input().length() ? sample.input().charAt(t) : paddingToken;
                inputData[t * batch + b] = vocabulary.indexOf(value);
            }
            for (int t = 0; t < targetTime; t++) {
                boolean valid = t < sample.target().length();
                char value = valid ? sample.target().charAt(t) : paddingToken;
                targetData[t * batch + b] = valid ? vocabulary.indexOf(value) : paddingIndex;
                maskData[t * batch + b] = (byte) (valid ? 1 : 0);
            }
        }

        return new SequenceBatch(
                new Tensor(inputData, new int[]{inputTime, batch}, List.of(), "text inputs", DataType.INT32),
                new Tensor(targetData, new int[]{targetTime, batch}, List.of(), "text targets", DataType.INT32),
                new Tensor(maskData, new int[]{targetTime, batch}, List.of(), "text mask", DataType.BOOL),
                lengths,
                true);
    }

    private static int maxInputLength(List<Sample<String, String>> samples) {
        int max = 0;
        for (Sample<String, String> sample : samples) {
            max = Math.max(max, sample.input().length());
        }
        return max;
    }

    private static int maxTargetLength(List<Sample<String, String>> samples) {
        int max = 0;
        for (Sample<String, String> sample : samples) {
            max = Math.max(max, sample.target().length());
        }
        return max;
    }
}
