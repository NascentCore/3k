package nascentcore.ai.modules.system.rest;

import cn.hutool.http.HttpRequest;
import cn.hutool.http.HttpResponse;
import cn.hutool.http.HttpUtil;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.config.ServiceProperties;
import nascentcore.ai.exception.BadRequestException;
import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import nascentcore.ai.modules.system.service.FileurlService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequiredArgsConstructor
@Api(tags = "模型：文件管理")
@RequestMapping("/api/info")
public class InfoUploadController {

    private final FileurlService fileurlService;
    private final ServiceProperties serviceProperties;
    @PostMapping(value = "/upload_status")
    @ApiOperation("上传模型数据文件 url")
    public ResponseEntity<Object> uploadStatus(@Validated @RequestBody String resources) {
        try {
            String url = serviceProperties.getServices().getScheduleService() + "info/upload_status";
            HttpRequest request = HttpUtil.createPost(url);
            request.body(resources);
            HttpResponse execute = request.execute();
            if (!execute.isOk()) {
                throw new BadRequestException("服务异常，调用调度模块upload_status接口异常");
            }
        } catch (Exception e) {
            throw new BadRequestException("服务异常，调用调度模块upload_status接口异常");
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
