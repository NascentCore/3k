package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@NoArgsConstructor
@Data
public class ResourceInfoDTO {
    @JsonProperty("cpod_id")
    private String cpodId;
    @JsonProperty("cpod_version")
    private String cpodVersion;
    @JsonProperty("gpu_summaries")
    private List<GpuSummariesDTO> gpuSummaries;
    @JsonProperty("nodes")
    private List<NodesDTO> nodes;
}
