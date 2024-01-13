package nascentcore.ai.modules.system.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import java.io.Serializable;
import java.sql.Timestamp;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author jim
 * @date 2023-9-27
 */
@Data
public class UserQueryCriteria implements Serializable {
    @ApiModelProperty(value = "用户ID")
    private Long userId;
}
