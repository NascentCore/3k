package nascentcore.ai.modules.system.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * @author jim
 * @date 2023-10-12
 **/
@Data
public class UserJobQueryCriteria{
    @ApiModelProperty(value = "ID")
    private Long jobId;
    @ApiModelProperty(value = "任务名称")
    private String jobName;
    @ApiModelProperty(value = "状态：1已下发、0未下发")
    private Integer obtainStatus;
    @ApiModelProperty(value = "用户ID")
    private Long userId;
    @ApiModelProperty(value = "cpod id")
    private String cpodId;
    @ApiModelProperty(value = "逻辑删除")
    private Integer deleted;
}
