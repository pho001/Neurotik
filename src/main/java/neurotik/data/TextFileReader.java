package neurotik.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class TextFileReader {
    private TextFileReader() {
    }

    public static DataSet<String> readLines(String fileName) {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = openReader(fileName)) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read text file: " + fileName, e);
        }
        return new DataSet<>(lines);
    }

    private static BufferedReader openReader(String fileName) throws IOException {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        InputStream resource = classLoader.getResourceAsStream(fileName);
        if (resource != null) {
            return new BufferedReader(new InputStreamReader(resource));
        }
        return Files.newBufferedReader(Path.of(fileName));
    }
}
