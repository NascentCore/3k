package nascentcore.ai.modules.system.domain.dto.file;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@NoArgsConstructor
@Data
public class ModelUrl {
    @JsonProperty("download_urls")
    private List<String> downloadUrls;
    @JsonProperty("job_name")
    private String jobName;
}
