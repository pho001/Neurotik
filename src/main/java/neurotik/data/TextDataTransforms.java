package neurotik.data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public final class TextDataTransforms {
    private TextDataTransforms() {
    }

    public static DataSet<String> charWindows(DataSet<String> lines, int contextLength, boolean fixedLengthPadding) {
        if (contextLength <= 0) {
            throw new IllegalArgumentException("Context length must be positive.");
        }
        List<String> windows = new ArrayList<>();
        for (String line : lines.data()) {
            String[] words = line.split(" ");
            for (String rawWord : words) {
                String word = ".".repeat(contextLength - 1) + rawWord + ".";
                if (fixedLengthPadding) {
                    for (int i = 0; i < word.length() - (contextLength - 1); i++) {
                        windows.add(word.substring(i, i + contextLength));
                    }
                } else {
                    for (int i = 1; i < word.length() - (contextLength - 1); i++) {
                        if (i <= contextLength - 1) {
                            windows.add(word.substring(contextLength - 1, i + contextLength));
                        } else {
                            windows.add(word.substring(i, i + contextLength));
                        }
                    }
                }
            }
        }
        return new DataSet<>(windows);
    }

    public static DataSet<String> sortByLengthDesc(DataSet<String> dataSet) {
        List<String> sorted = new ArrayList<>(dataSet.data());
        sorted.sort(Comparator.comparingInt(String::length).reversed());
        return new DataSet<>(sorted);
    }

    public static DataSet<String> sliceOffsets(DataSet<String> dataSet, int startOffset, int endOffset) {
        List<String> out = new ArrayList<>();
        for (String value : dataSet.data()) {
            int startIndex = startOffset;
            int endIndex = value.length() + endOffset;
            if (startIndex < 0 || startIndex > value.length()
                    || endIndex < 0 || endIndex > value.length()
                    || startIndex > endIndex) {
                throw new IllegalArgumentException("Invalid text offsets.");
            }
            out.add(value.substring(startIndex, endIndex));
        }
        return new DataSet<>(out);
    }

    public static DataSet<String> tail(DataSet<String> dataSet, int endOffset) {
        List<String> out = new ArrayList<>();
        for (String value : dataSet.data()) {
            int startIndex = value.length() + endOffset;
            int endIndex = value.length();
            if (startIndex < 0 || startIndex > value.length()
                    || endIndex < 0 || endIndex > value.length()
                    || startIndex > endIndex) {
                throw new IllegalArgumentException("Invalid text tail offset.");
            }
            out.add(value.substring(startIndex, endIndex));
        }
        return new DataSet<>(out);
    }

    public static DataSet<String> padToMaxLength(DataSet<String> dataSet, String paddingToken) {
        if (dataSet.size() == 0) {
            return dataSet;
        }
        int maxLength = 0;
        for (String value : dataSet.data()) {
            if (value.length() > maxLength) {
                maxLength = value.length();
            }
        }
        List<String> out = new ArrayList<>();
        for (String value : dataSet.data()) {
            out.add(value + paddingToken.repeat(maxLength - value.length()));
        }
        return new DataSet<>(out);
    }

    public static int[] lengths(DataSet<String> dataSet) {
        int[] lengths = new int[dataSet.size()];
        for (int i = 0; i < lengths.length; i++) {
            lengths[i] = dataSet.get(i).length();
        }
        return lengths;
    }

    public static int[] packedBatchSizes(int[] lengths) {
        int max = 0;
        for (int length : lengths) {
            if (length > max) {
                max = length;
            }
        }
        int[] packedSizes = new int[max];
        for (int i = 0; i < max; i++) {
            int count = 0;
            for (int length : lengths) {
                if (length > i) {
                    count++;
                }
            }
            packedSizes[i] = count;
        }
        return packedSizes;
    }

    public static String uniqueCharacters(DataSet<String> dataSet) {
        Set<Character> unique = new TreeSet<>();
        for (String value : dataSet.data()) {
            for (char c : value.toCharArray()) {
                unique.add(c);
            }
        }
        StringBuilder out = new StringBuilder();
        for (char character : unique) {
            out.append(character);
        }
        return out.toString();
    }
}
