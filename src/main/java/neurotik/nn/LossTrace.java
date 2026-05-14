package neurotik.nn;

import graph.execution.trace.CompileTrace;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.RunTrace;

import java.util.Comparator;

public record LossTrace(
        CompileTrace compile,
        PrepareTrace prepare,
        RunTrace run) {
    public String summary(int hotStepLimit) {
        int limit = Math.max(0, hotStepLimit);
        StringBuilder out = new StringBuilder();
        out.append("Compile: ")
                .append(compile.durationNs())
                .append(" ns, nodes=")
                .append(compile.totalNodeCount())
                .append(", backward=")
                .append(compile.supportsBackward())
                .append(System.lineSeparator());
        out.append("Prepare: ")
                .append(prepare.durationNs())
                .append(" ns, forwardSteps=")
                .append(prepare.forwardStepCount())
                .append(", backwardSteps=")
                .append(prepare.backwardStepCount())
                .append(System.lineSeparator());
        out.append("Run: ")
                .append(run.durationNs())
                .append(" ns, mode=")
                .append(run.mode())
                .append(", steps=")
                .append(run.steps().size())
                .append(", cpuMaterializations=")
                .append(run.cpuMaterializations().size())
                .append(System.lineSeparator());

        if (limit > 0) {
            out.append("Hot steps:").append(System.lineSeparator());
            run.steps().stream()
                    .sorted(Comparator.comparingLong(ExecutionStepTrace::durationNs).reversed())
                    .limit(limit)
                    .forEach(step -> out.append("  ")
                            .append(step.durationNs())
                            .append(" ns | ")
                            .append(step.opType())
                            .append(" | backend=")
                            .append(step.backend())
                            .append(" | kernel=")
                            .append(step.kernel())
                            .append(" | shape=")
                            .append(step.shape())
                            .append(" | label=")
                            .append(step.label())
                            .append(System.lineSeparator()));
        }
        return out.toString();
    }
}
