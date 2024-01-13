package nascentcore.ai.modules.system.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
public class PriceQueryCriteria{
    @ApiModelProperty(value = "ID")
    private Long priceId;
    @ApiModelProperty(value = "GPU型号")
    private String gpuProd;
}
