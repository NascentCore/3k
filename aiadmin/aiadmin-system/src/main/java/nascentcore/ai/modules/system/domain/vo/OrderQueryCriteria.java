package nascentcore.ai.modules.system.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
public class OrderQueryCriteria {
    @ApiModelProperty(value = "ID")
    private Long orderId;
    @ApiModelProperty(value = "用户ID")
    private Long userId;
    @ApiModelProperty(value = "job 名称")
    private String jobName;
    @ApiModelProperty(value = "订单号")
    private String tradeNo;
    @ApiModelProperty(value = "商户订单号")
    private String outTradeNo;
}
