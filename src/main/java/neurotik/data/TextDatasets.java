package neurotik.data;

import java.util.ArrayList;
import java.util.List;

public final class TextDatasets {
    private TextDatasets() {
    }

    public static SupervisedDataset<String, String> sequenceNextChar(DataSet<String> lines, int contextLength, String startToken) {
        validateTextWindow(contextLength, startToken);
        List<Sample<String, String>> samples = new ArrayList<>();
        for (String line : lines.data()) {
            for (String rawWord : line.trim().split("\\s+")) {
                if (rawWord.isEmpty()) {
                    continue;
                }
                String sequence = startToken + rawWord + startToken;
                if (sequence.length() <= contextLength) {
                    samples.add(new Sample<>(
                            sequence.substring(0, sequence.length() - 1),
                            sequence.substring(1)));
                    continue;
                }
                for (int start = 0; start <= sequence.length() - contextLength; start++) {
                    String window = sequence.substring(start, start + contextLength);
                    samples.add(new Sample<>(
                            window.substring(0, window.length() - 1),
                            window.substring(1)));
                }
            }
        }
        return new SupervisedDataset<>(samples);
    }

    public static SupervisedDataset<String, String> fixedNextChar(DataSet<String> lines, int contextLength, String startToken) {
        validateTextWindow(contextLength, startToken);
        List<Sample<String, String>> samples = new ArrayList<>();
        for (String line : lines.data()) {
            for (String rawWord : line.trim().split("\\s+")) {
                if (rawWord.isEmpty()) {
                    continue;
                }
                String word = startToken.repeat(contextLength - 1) + rawWord + startToken;
                for (int i = 0; i < word.length() - (contextLength - 1); i++) {
                    String window = word.substring(i, i + contextLength);
                    samples.add(new Sample<>(
                            window.substring(0, window.length() - 1),
                            window.substring(window.length() - 1)));
                }
            }
        }
        return new SupervisedDataset<>(samples);
    }

    private static void validateTextWindow(int contextLength, String startToken) {
        if (contextLength < 2) {
            throw new IllegalArgumentException("Context length must be at least 2.");
        }
        if (startToken.length() != 1) {
            throw new IllegalArgumentException("Text start token must be one character.");
        }
    }
}
