package nascentcore.ai.modules.system.domain;

import com.baomidou.mybatisplus.annotation.TableField;
import lombok.Data;
import cn.hutool.core.bean.BeanUtil;
import io.swagger.annotations.ApiModelProperty;
import cn.hutool.core.bean.copier.CopyOptions;
import java.sql.Timestamp;
import java.io.Serializable;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

/**
* @description /
* @author jimi
* @date 2023-10-23
**/
@Data
@TableName("sys_cpod_main")
public class CpodMain implements Serializable {

    @TableId(value = "main_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long mainId;

    @ApiModelProperty(value = "cpod id")
    private String cpodId;

    @ApiModelProperty(value = "pod 版本")
    private String cpodVersion;

    @ApiModelProperty(value = "gpu vendor")
    private String gpuVendor;

    @ApiModelProperty(value = "GPU型号")
    private String gpuProd;

    @ApiModelProperty(value = "GPU总数量")
    private Integer gpuTotal;

    @ApiModelProperty(value = "GPU可分配数量")
    private Integer gpuAllocatable;

    @ApiModelProperty(value = "创建日期")
    private Timestamp createTime;

    @ApiModelProperty(value = "更新时间")
    private Timestamp updateTime;

    @ApiModelProperty(value = "算力源注册ID")
    private String userId;

    @TableField(exist = false)
    @ApiModelProperty(value = "基础价格/hour/个")
    private Double amount;

    public void copy(CpodMain source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}
