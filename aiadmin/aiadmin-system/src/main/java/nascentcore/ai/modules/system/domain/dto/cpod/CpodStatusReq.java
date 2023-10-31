package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@NoArgsConstructor
@Data
public class CpodStatusReq {
    @JsonProperty("cpod_id")
    private String cpodId;
    @JsonProperty("job_status")
    private List<JobStatusDTO> jobStatus;
    @JsonProperty("resource_info")
    private ResourceInfoDTO resourceInfo;
    @JsonProperty("update_time")
    private String updateTime;
}
