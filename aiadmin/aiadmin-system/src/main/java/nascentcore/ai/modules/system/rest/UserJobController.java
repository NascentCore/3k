package nascentcore.ai.modules.system.rest;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.date.DateTime;
import com.alibaba.fastjson.JSON;
import nascentcore.ai.annotation.rest.AnonymousGetMapping;
import nascentcore.ai.annotation.rest.AnonymousPostMapping;
import nascentcore.ai.modules.system.domain.CpodMain;
import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.UserJobRes;
import nascentcore.ai.modules.system.domain.bean.Constants;
import nascentcore.ai.modules.system.domain.dto.cpod.CpodStatusReq;
import nascentcore.ai.modules.system.domain.dto.cpod.GpuSummariesDTO;
import nascentcore.ai.modules.system.domain.dto.cpod.JobStatusDTO;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import nascentcore.ai.modules.system.service.CpodMainService;
import nascentcore.ai.modules.system.service.JobManager;
import nascentcore.ai.modules.system.service.UserJobService;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.utils.SecurityUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import io.swagger.annotations.*;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.utils.PageResult;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static cn.hutool.core.util.IdUtil.randomUUID;


/**
 * @author jim
 * @date 2023-10-12
 **/
@RestController
@RequiredArgsConstructor
@Api(tags = "任务:  任务管理")
@RequestMapping("/api/userJob")
public class UserJobController {

    private final UserJobService userJobService;
    private final CpodMainService cpodMainService;
    private final JobManager jobManager;
    @GetMapping
    @ApiOperation("查询用户任务")
    public ResponseEntity<PageResult<UserJob>> queryUserJob(UserJobQueryCriteria criteria, Page<Object> page) {
        Long userId = SecurityUtils.getCurrentUserId();
        if (null != criteria) {
            criteria.setUserId(userId);
            return new ResponseEntity<>(userJobService.queryAll(criteria, page), HttpStatus.OK);
        } else {
            UserJobQueryCriteria criteria1 = new UserJobQueryCriteria();
            criteria1.setUserId(userId);
            return new ResponseEntity<>(userJobService.queryAll(criteria1, page), HttpStatus.OK);
        }
    }

    @PostMapping
    @ApiOperation("新增用户任务")
    public ResponseEntity<Object> createUserJob(@Validated @RequestBody UserJob resources){
        Long userId = SecurityUtils.getCurrentUserId();
        if(null != resources){
            resources.setUserId(userId);
            resources.setJobName("ai" + randomUUID());
            resources.setCreateTime(DateTime.now().toTimestamp());
            resources.setUpdateTime(DateTime.now().toTimestamp());
        }
        userJobService.create(resources);
        return new ResponseEntity<>(HttpStatus.CREATED);
    }

    @AnonymousGetMapping(value = "/cpod_jobs")
    @ApiOperation("获取未下发的用户新任务")
    public ResponseEntity<List<UserJobRes>> getNewJob(String cpodid) {
        List<CpodMain> cpodMains = cpodMainService.findByCpodId(cpodid);
        UserJobQueryCriteria criteria = new UserJobQueryCriteria();
        criteria.setObtainStatus(Constants.NEEDSEND);
        List<UserJob> userJobList = userJobService.queryAll(criteria);
        List<UserJobRes> userJobResList = new ArrayList<>();
        if (!cpodMains.isEmpty()) {
            for (CpodMain cpodMain : cpodMains) {
                for (UserJob job : userJobList) {
                    if (null == job.getCpodId()) {
                        int diff = cpodMain.getGpuAllocatable() - job.getGpuNumber();
                        if (diff >= 0 && cpodMain.getGpuProd().equals(job.getGpuType())) {
                            job.setCpodId(cpodid);
                            job.setUpdateTime(DateTime.now().toTimestamp());
                            UserJobRes userJobRes = new UserJobRes();
                            BeanUtil.copyProperties(job, userJobRes);
                            userJobResList.add(userJobRes);
                            cpodMain.setGpuAllocatable(diff);
                        }
                    } else if (job.getCpodId().equals(cpodMain.getCpodId())) {
                        if (cpodMain.getGpuProd().equals(job.getGpuType())) {
                            UserJobRes userJobRes = new UserJobRes();
                            BeanUtil.copyProperties(job, userJobRes);
                            userJobResList.add(userJobRes);
                        }
                    }
                }
            }
        }
        userJobService.save(userJobList);
        return new ResponseEntity<>(userJobResList, HttpStatus.OK);
    }

    @AnonymousPostMapping(value = "/cpod_status")
    @ApiOperation("上传三千平台信息")
    public ResponseEntity<Object> putPodStatus(@Validated @RequestBody String resources) {
        CpodStatusReq cpodStatusReq = JSON.parseObject(resources, CpodStatusReq.class);
        String cpodid = cpodStatusReq.getCpodId();
        List<JobStatusDTO> jobStatusDTOSList = cpodStatusReq.getJobStatus();
        jobManager.updateJobStatus(cpodid,jobStatusDTOSList);
        if (null != cpodid) {
            if (null != cpodStatusReq.getResourceInfo()) {
                for (GpuSummariesDTO gpuSummariesDTO : cpodStatusReq.getResourceInfo().getGpuSummaries()) {
                    CpodMainQueryCriteria criteria = new CpodMainQueryCriteria();
                    criteria.setCpodId(cpodid);
                    criteria.setGpuProd(gpuSummariesDTO.getProd());
                    List<CpodMain> cpodMainlist = cpodMainService.queryAll(criteria);
                    if (null != cpodMainlist && !cpodMainlist.isEmpty()) {
                        CpodMain cpodMain = cpodMainlist.get(0);
                        if (cpodMain.getGpuTotal() != gpuSummariesDTO.getTotal()) {
                            cpodMain.setGpuTotal(gpuSummariesDTO.getTotal());
                        }
                        if (cpodMain.getGpuAllocatable() != gpuSummariesDTO.getAllocatable()) {
                            cpodMain.setGpuAllocatable(gpuSummariesDTO.getAllocatable());
                        }
                        cpodMain.setUpdateTime(DateTime.now().toTimestamp());
                        cpodMainService.update(cpodMain);
                    } else {
                        CpodMain cpod = new CpodMain();
                        cpod.setCpodId(cpodid);
                        cpod.setCpodVersion(cpodStatusReq.getResourceInfo().getCpodVersion());
                        cpod.setGpuVendor(gpuSummariesDTO.getVendor());
                        cpod.setGpuProd(gpuSummariesDTO.getProd());
                        cpod.setGpuTotal(gpuSummariesDTO.getTotal());
                        cpod.setGpuAllocatable(gpuSummariesDTO.getAllocatable());
                        cpod.setCreateTime(DateTime.now().toTimestamp());
                        cpod.setUpdateTime(DateTime.now().toTimestamp());
                        cpodMainService.create(cpod);
                    }
                }
            }
        }
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @ApiOperation("返回支持的GPU类型")
    @GetMapping(value = "/getGpuType")
    public ResponseEntity<List<CpodMain>> queryAllGpuType(){
        return new ResponseEntity<>(cpodMainService.queryAllGpuType(),HttpStatus.OK);
    }

    @ApiOperation("删除任务")
    @DeleteMapping
    public ResponseEntity<Object> deleteJob(@RequestBody Set<Long> ids){
        userJobService.delete(ids);
        return new ResponseEntity<>(HttpStatus.OK);
    }
}