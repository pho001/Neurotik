package neurotik.data;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class TextVocabulary {
    private final char[] chars;
    private final Map<Character, Integer> indexes;

    public TextVocabulary(String characters) {
        this.chars = characters.toCharArray();
        this.indexes = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            indexes.put(chars[i], i);
        }
    }

    public static TextVocabulary from(DataSet<String> dataSet, String requiredTokens) {
        Set<Character> unique = new TreeSet<>();
        for (char c : requiredTokens.toCharArray()) {
            unique.add(c);
        }
        for (String value : dataSet.data()) {
            for (char c : value.toCharArray()) {
                unique.add(c);
            }
        }
        StringBuilder out = new StringBuilder();
        for (char c : unique) {
            out.append(c);
        }
        return new TextVocabulary(out.toString());
    }

    public int size() {
        return chars.length;
    }

    public int indexOf(char value) {
        Integer index = indexes.get(value);
        if (index == null) {
            throw new IllegalArgumentException("Character is not in vocabulary: " + value);
        }
        return index;
    }

    public char charAt(int index) {
        return chars[index];
    }
}
