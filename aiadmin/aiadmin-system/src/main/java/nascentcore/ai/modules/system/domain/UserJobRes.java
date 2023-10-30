package nascentcore.ai.modules.system.domain;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import java.io.Serializable;
@Data
public class UserJobRes implements Serializable {

    @ApiModelProperty(value = "任务名称")
    private String jobName;

    @ApiModelProperty(value = "用户任务ID")
    private Long jobId;

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
}

