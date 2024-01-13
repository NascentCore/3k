package nascentcore.ai.modules.system.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
* @author jimi
* @date 2023-10-23
**/
@Data
public class CpodMainQueryCriteria {
    @ApiModelProperty(value = "cpod id")
    private String cpodId;
    @ApiModelProperty(value = "GPU型号")
    private String gpuProd;
    @ApiModelProperty(value = "GPU可分配数量")
    private Integer gpuAllocatable;
    @ApiModelProperty(value = "算力源注册ID")
    private String userId;
}