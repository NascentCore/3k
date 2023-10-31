package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class GpuStateDTO {
    @JsonProperty("status")
    private String status;
    @JsonProperty("allocated")
    private boolean allocated;
    @JsonProperty("mem_usage")
    private int memUsage;
    @JsonProperty("gpu_usage")
    private int gpuUsage;
    @JsonProperty("power")
    private int power;
    @JsonProperty("temp")
    private int temp;
}
