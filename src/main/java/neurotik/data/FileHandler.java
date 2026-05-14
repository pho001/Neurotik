package neurotik.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class FileHandler {
    private final String fileName;

    public FileHandler(String fileName){
        this.fileName=fileName;
    }

    public List<String> ReadFileLines() {
        List <String> lines=new ArrayList<>();

        try (BufferedReader br = openReader()) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return lines;
    }

    private BufferedReader openReader() throws IOException {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        InputStream resource = classLoader.getResourceAsStream(fileName);
        if (resource != null) {
            return new BufferedReader(new InputStreamReader(resource));
        }

        return Files.newBufferedReader(Path.of(fileName));
    }
}
