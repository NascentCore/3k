package nascentcore.ai.modules.system.domain.vo;

import lombok.Data;

/**
* @author jimi
* @date 2023-10-23
**/
@Data
public class CpodMainQueryCriteria{
    private String cpodId;
    private String gpuProd;
    private Integer gpuAllocatable;
    private String userId;
}