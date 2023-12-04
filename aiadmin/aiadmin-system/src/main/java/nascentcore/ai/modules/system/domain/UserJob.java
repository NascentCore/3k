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

/**
 * @description /
 * @author jim
 * @date 2023-10-12
 **/
@Data
@TableName("sys_user_job")
public class UserJob implements Serializable {

    @TableId(value = "job_id", type = IdType.AUTO)
    @ApiModelProperty(value = "ID")
    private Long jobId;

    @ApiModelProperty(value = "任务名称")
    private String jobName;

    @ApiModelProperty(value = "GPU数量")
    private Integer gpuNumber;

    @ApiModelProperty(value = "GPU型号")
    private String gpuType;

    @ApiModelProperty(value = "cktp路径")
    private String ckptPath;

    @ApiModelProperty(value = "cktp容量")
    private String ckptVol;

    @ApiModelProperty(value = "save model路径")
    private String modelPath;

    @ApiModelProperty(value = "save model容量")
    private String modelVol;

    @ApiModelProperty(value = "镜像路径")
    private String imagePath;

    @ApiModelProperty(value = "HF公开训练数据URL")
    private String hfUrl;

    @ApiModelProperty(value = "挂载路径")
    private String datasetPath;

    @ApiModelProperty(value = "任务类型 mpi")
    private String jobType;

    @ApiModelProperty(value = "0 自然终止 1设定时长")
    private Integer stopType;

    @ApiModelProperty(value = "设定时常以小时为单位")
    private Integer stopTime;

    @ApiModelProperty(value = "状态：1暂停、0运行、2完成")
    private Integer workStatus;

    @ApiModelProperty(value = "状态：1已下发、0未下发")
    private Integer obtainStatus;

    @ApiModelProperty(value = "创建日期")
    private Timestamp createTime;

    @ApiModelProperty(value = "更新时间")
    private Timestamp updateTime;

    @ApiModelProperty(value = "用户ID")
    private Long userId;

    @ApiModelProperty(value = "cpod id")
    private String cpodId;

    @ApiModelProperty(value = "模型启动命令")
    private String runCommand;

    @ApiModelProperty(value = "第三方回调接口url")
    private String callbackUrl;

    @ApiModelProperty(value = "挂载路径名称")
    private String datasetName;

    @ApiModelProperty(value = "模型基座名称")
    private String pretrainedModelName;

    @ApiModelProperty(value = "模型基座对应路径")
    private String pretrainedModelPath;
    public void copy(UserJob source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}
