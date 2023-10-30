package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
public class GpuInfoDTO {
    @JsonProperty("status")
    private String status;
    @JsonProperty("vendor")
    private String vendor;
    @JsonProperty("prod")
    private String prod;
    @JsonProperty("driver")
    private String driver;
    @JsonProperty("cuda")
    private String cuda;
    @JsonProperty("mem_size")
    private int memSize;
}
