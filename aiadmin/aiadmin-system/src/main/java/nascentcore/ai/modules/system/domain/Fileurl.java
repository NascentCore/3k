package nascentcore.ai.modules.system.domain;

import cn.hutool.core.bean.BeanUtil;
import io.swagger.annotations.ApiModelProperty;
import cn.hutool.core.bean.copier.CopyOptions;
import java.sql.Timestamp;
import java.io.Serializable;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;

import javax.validation.constraints.NotBlank;

@Getter
@Setter
@TableName("sys_fileurl")
public class Fileurl implements Serializable {

    @TableId(value = "file_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long fileId;

    @NotBlank
    @ApiModelProperty(value = "job 名称")
    private String jobName;

    @ApiModelProperty(value = "文件链接")
    private String fileUrl;

    @ApiModelProperty(value = "创建日期")
    private Timestamp createTime;

    @ApiModelProperty(value = "更新时间")
    private Timestamp updateTime;

    public void copy(Fileurl source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}

