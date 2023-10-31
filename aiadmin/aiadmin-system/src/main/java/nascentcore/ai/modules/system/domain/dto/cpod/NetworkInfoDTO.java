package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class NetworkInfoDTO {
    @JsonProperty("type")
    private String type;
    @JsonProperty("throughput")
    private int throughput;
}
