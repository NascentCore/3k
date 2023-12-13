package nascentcore.ai.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import javax.validation.constraints.NotBlank;
import java.io.Serializable;

@Data
@TableName("tool_alipay_config")
public class AlipayConfig implements Serializable {

    @TableId("config_id")
    private Long id;

    @NotBlank
    @ApiModelProperty(value = "应用ID")
    private String appId;

    @NotBlank
    @ApiModelProperty(value = "商户私钥")
    private String privateKey;

    @NotBlank
    @ApiModelProperty(value = "支付宝公钥")
    private String publicKey;

    @ApiModelProperty(value = "签名方式")
    private String signType = "RSA2";

    @ApiModelProperty(value = "支付宝开放安全地址", hidden = true)
    private String gatewayUrl = "https://openapi.alipaydev.com/gateway.do";

    @ApiModelProperty(value = "编码", hidden = true)
    private String charset = "utf-8";

    @NotBlank
    @ApiModelProperty(value = "异步通知地址")
    private String notifyUrl;

    @NotBlank
    @ApiModelProperty(value = "订单完成后返回的页面")
    private String returnUrl;

    @ApiModelProperty(value = "类型")
    private String format = "JSON";

    @NotBlank
    @ApiModelProperty(value = "商户号")
    private String sysServiceProviderId;

}
