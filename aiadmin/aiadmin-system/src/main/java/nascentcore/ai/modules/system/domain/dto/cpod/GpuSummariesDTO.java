package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class GpuSummariesDTO {
    @JsonProperty("vendor")
    private String vendor;
    @JsonProperty("prod")
    private String prod;
    @JsonProperty("total")
    private int total;
    @JsonProperty("allocatable")
    private int allocatable;
}
