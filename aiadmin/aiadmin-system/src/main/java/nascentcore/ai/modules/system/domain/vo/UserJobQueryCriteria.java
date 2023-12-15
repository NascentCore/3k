package nascentcore.ai.modules.system.domain.vo;

import lombok.Data;

/**
 * @author jim
 * @date 2023-10-12
 **/
@Data
public class UserJobQueryCriteria{
    private Long jobId;
    private String jobName;
    private Integer obtainStatus;
    private Long userId;
    private String cpodId;
    private Integer deleted;
}
