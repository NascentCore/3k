package nascentcore.ai.modules.system.domain.dto.cpod;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@NoArgsConstructor
@Data
public class NodesDTO {
    @JsonProperty("name")
    private String name;
    @JsonProperty("status")
    private String status;
    @JsonProperty("arch")
    private String arch;
    @JsonProperty("kernel_version")
    private String kernelVersion;
    @JsonProperty("linux_dist")
    private String linuxDist;
    @JsonProperty("gpu_info")
    private GpuInfoDTO gpuInfo;
    @JsonProperty("gpu_total")
    private int gpuTotal;
    @JsonProperty("gpu_allocatable")
    private int gpuAllocatable;
    @JsonProperty("gpu_state")
    private List<GpuStateDTO> gpuState;
    @JsonProperty("cpu_info")
    private CpuInfoDTO cpuInfo;
    @JsonProperty("mem_info")
    private MemInfoDTO memInfo;
    @JsonProperty("disk_info")
    private DiskInfoDTO diskInfo;
    @JsonProperty("network_info")
    private NetworkInfoDTO networkInfo;
}
