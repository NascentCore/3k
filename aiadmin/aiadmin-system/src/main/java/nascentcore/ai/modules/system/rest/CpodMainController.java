package nascentcore.ai.modules.system.rest;

import nascentcore.ai.modules.system.domain.CpodMain;
import nascentcore.ai.modules.system.service.CpodMainService;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.utils.SecurityUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import io.swagger.annotations.*;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.utils.PageResult;

@RestController
@RequiredArgsConstructor
@Api(tags = "cpod：信息管理")
@RequestMapping("/api/cpod")
public class CpodMainController {

    private final CpodMainService cpodMainService;

    @GetMapping
    @ApiOperation("查询cpod信息")
    public ResponseEntity<PageResult<CpodMain>> queryCpodMain(CpodMainQueryCriteria criteria, Page<Object> page){
        Long userId = SecurityUtils.getCurrentUserId();
        CpodMainQueryCriteria cri = new CpodMainQueryCriteria();
        cri.setUserId(String.valueOf(userId));
        return new ResponseEntity<>(cpodMainService.queryAll(cri,page),HttpStatus.OK);
    }

}

