package portalsync

import (
	"sxwl/3k/pkg/communication"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/log"
	"sync"
	"time"
)

type itemToDelete struct {
	jobName    string
	JobType    state.JobType
	deleteTime time.Time
}

type jobBuffer struct {
	m  map[string]job.Job
	mu *sync.RWMutex
}

// 记录创建失败的任务，方便将信息上报
var createFailedJobs jobBuffer = jobBuffer{map[string]job.Job{}, new(sync.RWMutex)}

func GetCreateFailedJobs() []job.Job {
	res := []job.Job{}
	createFailedJobs.mu.RLock()
	defer createFailedJobs.mu.RUnlock()
	for _, v := range createFailedJobs.m {
		res = append(res, v)
	}
	return res
}

func addCreateFailedJob(j job.Job) {
	if _, ok := createFailedJobs.m[j.JobID]; ok {
		return
	}
	createFailedJobs.mu.Lock()
	defer createFailedJobs.mu.Unlock()
	createFailedJobs.m[j.JobID] = j
}

// 如果任务创建成功了，将其从失败任务列表中删除
func deleteCreateFailedJob(j string) {
	if _, ok := createFailedJobs.m[j]; !ok {
		return
	}
	createFailedJobs.mu.Lock()
	defer createFailedJobs.mu.Unlock()
	delete(createFailedJobs.m, j)
}

func StartPortalSync() {
	deleteBuffer := map[string]itemToDelete{}
	for {
		time.Sleep(time.Second * 10)
		jobTypeMap := map[string]state.JobType{}
		//jobSet A
		jobs, err := communication.GetJobs(config.CPOD_ID)
		if err != nil {
			log.SLogger.Errorw("get jobs from portal error", "error", err)
			continue
		} else {
			log.SLogger.Info("get jobs from portal succeed")
		}
		jobSetA := map[string]struct{}{}
		for _, j := range jobs {
			jobSetA[j.JobID] = struct{}{}
			jobTypeMap[j.JobID] = j.JobType
		}

		//get all jobs in cpod , jobSet B
		jobStates, err := job.GetJobStates()
		if err != nil {
			log.SLogger.Errorw("get jobs state error", "error", err)
			continue
		}
		jobSetB := map[string]struct{}{}
		for _, j := range jobStates {
			jobSetB[j.Name] = struct{}{}
			jobTypeMap[j.Name] = j.JobType
		}
		// TODO: Use set difference to calculate the jobs in A & B that should be removed or created.
		//delete all jobs in B and not in A
		for _, jobState := range jobStates {
			//if !jobState.JobStatus.NoMoreChange() {
			if _, ok := jobSetA[jobState.Name]; !ok {
				//keep the failed mpijobs for 24 hours for error detection)
				if jobState.JobStatus == state.JobStatusCreateFailed || jobState.JobStatus == state.JobStatusFailed {
					// delete after 24 hours
					if _, ok := deleteBuffer[jobState.Name]; !ok {
						deleteBuffer[jobState.Name] = itemToDelete{jobState.Name, jobState.JobType, time.Now().Add(time.Hour * 24)}
						log.SLogger.Infow("queued for delete job", "jobname", jobState.Name)
					}
				} else if jobState.JobStatus == state.JobStatusSucceed {
					// just delete the mpijob , keep the uploader job and pvc
					// when status is succeed , either the job is succeeded or the job is time up
					job.DeleteJob(jobState.Name, jobState.JobType, false)
				} else { // triggered by user , delete all related resources(uploader job and pvcs)
					job.DeleteJob(jobState.Name, jobState.JobType, true)
				}
			}
		}
		//create all jobs in A but not in B
		for _, job := range jobs {
			//if job not in B
			if _, ok := jobSetB[job.JobID]; !ok {
				//if create failed , just skip , try nexttime
				err := job.Run()
				if err != nil {
					addCreateFailedJob(job)
					log.SLogger.Errorw("Job run failed",
						"job", job,
						"error", err)
				} else {
					deleteCreateFailedJob(job.JobID)
					log.SLogger.Infow("Job created",
						"job", job)
				}
			}
		}
		for jobName, deleteItem := range deleteBuffer {
			if deleteItem.deleteTime.Before(time.Now()) {
				job.DeleteJob(jobName, deleteItem.JobType, true)
				delete(deleteBuffer, jobName)
			}
		}
	}
}
