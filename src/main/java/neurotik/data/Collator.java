package neurotik.data;

import java.util.List;

public interface Collator<S, B extends Batch> {
    B collate(List<S> samples);
}
