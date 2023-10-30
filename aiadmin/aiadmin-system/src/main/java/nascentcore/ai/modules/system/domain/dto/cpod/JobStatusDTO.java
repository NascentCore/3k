package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class JobStatusDTO {
    @JsonProperty("name")
    private String name;
    @JsonProperty("namespace")
    private String namespace;
    @JsonProperty("job_status")
    private String jobStatus;
    @JsonProperty("extension")
    private Object extension;
}
