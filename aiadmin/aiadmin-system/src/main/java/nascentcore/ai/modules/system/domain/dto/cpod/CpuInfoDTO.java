package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class CpuInfoDTO {
    @JsonProperty("cores")
    private int cores;
    @JsonProperty("usage")
    private int usage;
}
