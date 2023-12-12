package nascentcore.ai.modules.system.rest;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;
import nascentcore.ai.modules.system.domain.CpodMain;
import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.bean.Constants;
import nascentcore.ai.modules.system.domain.dto.cpod.CpodStatusReq;
import nascentcore.ai.modules.system.domain.dto.cpod.GpuSummariesDTO;
import nascentcore.ai.modules.system.domain.dto.cpod.JobStatusDTO;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import nascentcore.ai.modules.system.service.CpodMainService;
import nascentcore.ai.modules.system.service.FileurlService;
import nascentcore.ai.modules.system.service.JobManager;
import nascentcore.ai.modules.system.service.UserJobService;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.utils.DateUtil;
import nascentcore.ai.utils.SecurityUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import io.swagger.annotations.*;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import nascentcore.ai.utils.PageResult;

import java.util.*;

import static cn.hutool.core.util.IdUtil.randomUUID;


/**
 * @author jim
 * @date 2023-10-12
 **/
@RestController
@RequiredArgsConstructor
@Api(tags = "任务：任务管理")
@RequestMapping("/api/userJob")
public class UserJobController {

    private final UserJobService userJobService;
    private final CpodMainService cpodMainService;
    private final JobManager jobManager;
    private final FileurlService fileurlService;
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

    @PostMapping(value = "/job_status")
    @ApiOperation("查询用户任务状态")
    public ResponseEntity<Object> queryUserJobStatus(@Validated @RequestBody String resources) {
        JSONObject obj = JSON.parseObject(resources);
        String jobId = obj.getString("job_id");
        Map<String, Object> objmap = new HashMap<String, Object>();
        UserJobQueryCriteria criteria = new UserJobQueryCriteria();
        criteria.setJobName(jobId);
        List<UserJob> userJobList = userJobService.queryAll(criteria);
        if (null != userJobList && !userJobList.isEmpty()) {
            int workstatus = userJobList.get(0).getWorkStatus();
            if(Constants.WORKER_STATUS_URL_SUCCESS == workstatus){
                objmap.put("status","success");
                FileurlQueryCriteria fileurlQueryCriteria = new FileurlQueryCriteria();
                fileurlQueryCriteria.setJobName(jobId);
                List<Fileurl> fileurlList = fileurlService.queryAll(fileurlQueryCriteria);
                if (null != fileurlList && !fileurlList.isEmpty()) {
                    objmap.put("url", fileurlList.get(0).getFileUrl());
                }
            }else if(Constants.WORKER_STATUS_FIAL == workstatus){
                objmap.put("status","fail");
            }else {
                objmap.put("status","working");
            }
        }
        return new ResponseEntity<>(objmap,HttpStatus.OK);
    }

    @PostMapping
    @ApiOperation("新增用户任务")
    public ResponseEntity<Object> createUserJob(@Validated @RequestBody String resources){
        Long userId = SecurityUtils.getCurrentUserId();
        UserJob userJob = JSON.parseObject(resources,UserJob.class);
        Map<String, Object> obj = new HashMap<String, Object>();
        if(null != userJob){
            userJob.setUserId(userId);
            userJob.setJobName("ai" + randomUUID());
            userJob.setCreateTime(DateUtil.getUTCTimeStamp());
            userJob.setUpdateTime(DateUtil.getUTCTimeStamp());
            obj.put("job_id",userJob.getJobName());
            JSONObject jsonObjectobj = JSON.parseObject(resources);
            jsonObjectobj.put("jobName",userJob.getJobName());
            jsonObjectobj.put("modelVol",Integer.valueOf(userJob.getModelVol()));
            jsonObjectobj.put("ckptVol",Integer.valueOf(userJob.getCkptVol()));
            jsonObjectobj.put("stopType", userJob.getStopType());
            String output = JSON.toJSONString(jsonObjectobj, SerializerFeature.PrettyFormat);
            userJob.setJsonAll(output);
        }
        userJobService.create(userJob);
        return new ResponseEntity<>(obj,HttpStatus.OK);
    }

    @GetMapping(value = "/cpod_jobs")
    @ApiOperation("获取未下发的用户新任务")
    public ResponseEntity<List<JSONObject>> getNewJob(String cpodid) {
        List<CpodMain> cpodMains = cpodMainService.findByCpodId(cpodid);
        UserJobQueryCriteria criteria = new UserJobQueryCriteria();
        criteria.setObtainStatus(Constants.NEEDSEND);
        List<UserJob> userJobList = userJobService.queryAll(criteria);
        List<JSONObject> userJobResList = new ArrayList<>();
        if (!cpodMains.isEmpty()) {
            for (CpodMain cpodMain : cpodMains) {
                for (UserJob job : userJobList) {
                    if (null == job.getCpodId()) {
                        int diff = cpodMain.getGpuAllocatable() - job.getGpuNumber();
                        if (diff >= 0 && cpodMain.getGpuProd().equals(job.getGpuType())) {
                            job.setCpodId(cpodid);
                            job.setUpdateTime(DateUtil.getUTCTimeStamp());
                            if(null != job.getJsonAll()){
                                userJobResList.add(JSON.parseObject(job.getJsonAll()));
                            }
                            cpodMain.setGpuAllocatable(diff);
                        }
                    } else if (job.getCpodId().equals(cpodMain.getCpodId())) {
                        if (cpodMain.getGpuProd().equals(job.getGpuType())) {
                            if(null != job.getJsonAll()){
                                userJobResList.add(JSON.parseObject(job.getJsonAll()));
                            }
                        }
                    }
                }
            }
        }
        userJobService.save(userJobList);
        return new ResponseEntity<>(userJobResList, HttpStatus.OK);
    }

    @PostMapping(value = "/cpod_status")
    @ApiOperation("上传三千平台信息")
    public ResponseEntity<Object> putPodStatus(@Validated @RequestBody String resources) {
        Long userId = SecurityUtils.getCurrentUserId();
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
                        cpodMain.setUserId(String.valueOf(userId));
                        cpodMain.setUpdateTime(DateUtil.getUTCTimeStamp());
                        cpodMainService.update(cpodMain);
                    } else {
                        CpodMain cpod = new CpodMain();
                        cpod.setCpodId(cpodid);
                        cpod.setCpodVersion(cpodStatusReq.getResourceInfo().getCpodVersion());
                        cpod.setGpuVendor(gpuSummariesDTO.getVendor());
                        cpod.setGpuProd(gpuSummariesDTO.getProd());
                        cpod.setGpuTotal(gpuSummariesDTO.getTotal());
                        cpod.setGpuAllocatable(gpuSummariesDTO.getAllocatable());
                        cpod.setUserId(String.valueOf(userId));
                        cpod.setCreateTime(DateUtil.getUTCTimeStamp());
                        cpod.setUpdateTime(DateUtil.getUTCTimeStamp());
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

    @ApiOperation("终止任务")
    @PostMapping(value = "/job_del")
    public ResponseEntity<Object> logicDeleteJob(@RequestBody String body){
        JSONObject obj = JSON.parseObject(body);
        String jobId = obj.getString("job_id");
        userJobService.deletebyName(jobId);
        return new ResponseEntity<>(HttpStatus.OK);
    }
}
