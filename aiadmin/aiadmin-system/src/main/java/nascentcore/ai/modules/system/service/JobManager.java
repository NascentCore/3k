package nascentcore.ai.modules.system.service;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.date.DateTime;
import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.bean.Constants;
import nascentcore.ai.modules.system.domain.dto.cpod.JobStatusDTO;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import nascentcore.ai.modules.system.thread.HttpSenddateThread;
import nascentcore.ai.utils.DateUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import nascentcore.ai.utils.StringUtils;
import java.util.ArrayList;
import java.util.List;

@Component
public class JobManager {
    @Autowired
    public UserJobService userJobService;
    public void updateJobStatus(String cpodid, List<JobStatusDTO> jobStatusDTOSList) {
        UserJobQueryCriteria criteria = new UserJobQueryCriteria();
        criteria.setCpodId(cpodid);
        List<UserJob> userJobList = userJobService.queryAll(criteria);
        List<UserJob> userJobListtmp = new ArrayList<>();
        if (null != jobStatusDTOSList && !jobStatusDTOSList.isEmpty()) {
            for (JobStatusDTO jobStatusDTO : jobStatusDTOSList) {
                for (UserJob userJob : userJobList) {
                    if (jobStatusDTO.getName().equals(userJob.getJobName())) {
                        String status = jobStatusDTO.getJobStatus();
                        if (("createfailed".equals(status) || "failed".equals(status)) && Constants.WORKER_STATUS_FIAL != userJob.getWorkStatus()) {
                            userJob.setWorkStatus(Constants.WORKER_STATUS_FIAL);
                            userJob.setObtainStatus(Constants.NOTNEEDSEND);
                            userJob.setUpdateTime(DateUtil.getUTCTimeStamp());
                            UserJob job = new UserJob();
                            BeanUtil.copyProperties(userJob, job);
                            userJobListtmp.add(job);
                            if(!StringUtils.isEmpty(userJob.getCallbackUrl())){
                                HttpSenddateThread.add(job);
                            }
                        } else if ("succeeded".equals(status) && Constants.WORKER_STATUS_SUCCESS != userJob.getWorkStatus()) {
                            userJob.setWorkStatus(Constants.WORKER_STATUS_SUCCESS);
                            userJob.setObtainStatus(Constants.NOTNEEDSEND);
                            userJob.setUpdateTime(DateUtil.getUTCTimeStamp());
                            UserJob job = new UserJob();
                            BeanUtil.copyProperties(userJob, job);
                            userJobListtmp.add(job);
                        } else if ("modeluploaded".equals(status) && Constants.WORKER_STATUS_URL_SUCCESS != userJob.getWorkStatus()) {
                            userJob.setWorkStatus(Constants.WORKER_STATUS_URL_SUCCESS);
                            userJob.setObtainStatus(Constants.NOTNEEDSEND);
                            UserJob job = new UserJob();
                            BeanUtil.copyProperties(userJob, job);
                            userJobListtmp.add(job);
                        }
                    }
                }
            }
        }
        if (!userJobListtmp.isEmpty()) {
            userJobService.save(userJobListtmp);
        }
    }
}
