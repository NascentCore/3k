/*
*  Copyright 2019-2023 Zheng Jie
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
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

    public void copy(CpodMain source){
        BeanUtil.copyProperties(source,this, CopyOptions.create().setIgnoreNullValue(true));
    }
}
