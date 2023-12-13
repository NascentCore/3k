package nascentcore.ai.modules.system.domain;

import lombok.Data;
import cn.hutool.core.bean.BeanUtil;
import io.swagger.annotations.ApiModelProperty;
import cn.hutool.core.bean.copier.CopyOptions;
import java.io.Serializable;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

@Data
@TableName("sys_config")
public class Config implements Serializable {

    @TableId(value = "config_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long configId;

    @ApiModelProperty(value = "系统配置名")
    private String keyName;

    @ApiModelProperty(value = "系统配置值")
    private String keyValue;

    public void copy(Config source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}
