package nascentcore.ai.modules.system.rest;

import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.date.DateTime;
import com.alibaba.fastjson.JSON;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.dto.file.ModelUrl;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import nascentcore.ai.modules.system.service.FileurlService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequiredArgsConstructor
@Api(tags = "模型：文件管理")
@RequestMapping("/api/info")
public class InfoUploadController {

    private final FileurlService fileurlService;
    @PostMapping(value = "/upload_status")
    @ApiOperation("上传模型数据文件 url")
    public ResponseEntity<Object> uploadStatus(@Validated @RequestBody String resources) {
        ModelUrl modelUrl = JSON.parseObject(resources, ModelUrl.class);
        List<Fileurl> fileurlList = new ArrayList<>();
        if (null != modelUrl && CollectionUtil.isNotEmpty(modelUrl.getDownloadUrls())) {
            for (String it : modelUrl.getDownloadUrls()) {
                Fileurl fileurl = new Fileurl();
                fileurl.setJobName(modelUrl.getJobName());
                fileurl.setFileUrl(it);
                fileurl.setCreateTime(DateTime.now().toTimestamp());
                fileurl.setUpdateTime(DateTime.now().toTimestamp());
                fileurlList.add(fileurl);
            }
            fileurlService.save(fileurlList);
        }
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @GetMapping(value = "/model_urls")
    @ApiOperation("获取模型数据下载链接")
    public ResponseEntity<List<Fileurl>> getModelUrls(String jobName) {
        FileurlQueryCriteria fileurlQueryCriteria = new FileurlQueryCriteria();
        fileurlQueryCriteria.setJobName(jobName);
        List<Fileurl> fileurlList = fileurlService.queryAll(fileurlQueryCriteria);
        return new ResponseEntity<>(fileurlList, HttpStatus.OK);
    }
}