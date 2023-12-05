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
    private Integer ckptVol;

    @ApiModelProperty(value = "save model路径")
    private String modelPath;

    @ApiModelProperty(value = "save model容量")
    private Integer modelVol;

    @ApiModelProperty(value = "镜像路径")
    private String imagePath;

    @ApiModelProperty(value = "HF公开训练数据URL")
    private String hfUrl;

    @ApiModelProperty(value = "挂载路径")
    private String datasetPath;

    @ApiModelProperty(value = "任务类型 MPI;Pytorch;TensorFlow;GeneralJob")
    private String jobType;

    @ApiModelProperty(value = "0 自然终止 1设定时长")
    private Integer stopType;

    @ApiModelProperty(value = "设定时常以分钟为单位")
    private Integer stopTime;

    @ApiModelProperty(value = "模型启动命令")
    private String runCommand;

    @ApiModelProperty(value = "挂载路径名称")
    private String datasetName;

    @ApiModelProperty(value = "模型基座名称")
    private String pretrainedModelName;

    @ApiModelProperty(value = "模型基座对应路径")
    private String pretrainedModelPath;
}

