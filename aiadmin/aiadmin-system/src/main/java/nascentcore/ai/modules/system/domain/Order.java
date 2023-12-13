package nascentcore.ai.modules.system.domain;

import lombok.Data;
import cn.hutool.core.bean.BeanUtil;
import io.swagger.annotations.ApiModelProperty;
import cn.hutool.core.bean.copier.CopyOptions;
import java.sql.Timestamp;
import java.io.Serializable;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

@Data
@TableName("sys_order")
public class Order implements Serializable {

    @TableId(value = "order_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long orderId;

    @ApiModelProperty(value = "用户ID")
    private Long userId;

    @ApiModelProperty(value = "job 名称")
    private String jobName;

    @ApiModelProperty(value = "订单号")
    private String tradeNo;

    @ApiModelProperty(value = "商品描述")
    private String body;

    @ApiModelProperty(value = "商品名称")
    private String subject;

    @ApiModelProperty(value = "支付金额")
    private Double totalAmount;

    @ApiModelProperty(value = "状态 0未付款 1已付款")
    private Integer status;

    @ApiModelProperty(value = "创建日期")
    private Timestamp createTime;

    @ApiModelProperty(value = "更新时间")
    private Timestamp updateTime;

    @ApiModelProperty(value = "商户订单号")
    private String outTradeNo;

    public void copy(Order source) {
        BeanUtil.copyProperties(source, this, CopyOptions.create().setIgnoreNullValue(true));
    }
}
