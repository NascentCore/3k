package nascentcore.ai.modules.system.domain.dto.job;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class EnvDTO {
    @JsonProperty("modihandOpenNodeToken")
    private String modihandOpenNodeToken;
}
