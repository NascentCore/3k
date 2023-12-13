package nascentcore.ai.modules.system.domain;

import lombok.Data;
import java.io.Serializable;

@Data
public class AliPayRes implements Serializable {
    /** （必填）商品描述 */
    private String body;

    /** （必填）商品名称 */
    private String subject;

    /** （必填）价格 */
    private String totalAmount;

    /** 订单状态,1已支付，0未支付， */
    private String state;

    /** 订单二维码 */
    private String qrCode;
}
