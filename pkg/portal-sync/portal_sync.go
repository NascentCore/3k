package portalsync

import (
	"sxwl/3k/pkg/communication"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/log"
	"time"
)

type itemToDelete struct {
	jobName    string
	JobType    state.JobType
	deleteTime time.Time
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
					log.SLogger.Errorw("Job run failed",
						"job", job,
						"error", err)
				} else {
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
