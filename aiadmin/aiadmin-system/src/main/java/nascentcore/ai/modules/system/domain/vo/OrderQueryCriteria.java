package nascentcore.ai.modules.system.domain.vo;

import lombok.Data;

@Data
public class OrderQueryCriteria{
    private Long orderId;
    private Long userId;
    private String jobName;
    private String tradeNo;
    private String outTradeNo;
}
