package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class DiskInfoDTO {
    @JsonProperty("size")
    private int size;
    @JsonProperty("usage")
    private int usage;
}
