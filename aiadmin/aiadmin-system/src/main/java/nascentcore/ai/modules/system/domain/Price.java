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
@TableName("sys_price")
public class Price implements Serializable {

    @TableId(value = "price_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long priceId;

    @ApiModelProperty(value = "GPU型号")
    private String gpuProd;

    @ApiModelProperty(value = "基础价格/hour/个")
    private Double amount;

    @ApiModelProperty(value = "创建日期")
    private Timestamp createTime;

    public void copy(Price source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}

