package nascentcore.ai.modules.system.domain.vo;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
@Data
public class FileurlQueryCriteria {
    @ApiModelProperty(value = "job 名称")
    private String jobName;
}
