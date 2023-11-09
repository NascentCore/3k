package main

// NO_TEST_NEEDED

import (
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/communication"
	"sxwl/3k/manager/pkg/config"
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/resource"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
)

func main() {
	clientgo.InitClient()

	// upload resource & task info , also as heartbeat , indicate this cpod is alive and ready for tasks
	var done chan struct{}
	startUploadInfo(done)
	// clean mpijob job and pvc in specified namespace
	startCleanUp(done, config.CPOD_NAMESPACE)
	// get tasks , then run them !!!
	//wait to delete
	deleteBuffer := map[string]time.Time{}
	for {
		time.Sleep(time.Second * 10)
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
		}
		//delete all jobs in B and not in A
		for _, jobState := range jobStates {
			//if !jobState.JobStatus.NoMoreChange() {
			if _, ok := jobSetA[jobState.Name]; !ok {
				//keep the failed mpijobs for 24 hours for error detection)
				if jobState.JobStatus == state.JobStatusCreateFailed || jobState.JobStatus == state.JobStatusFailed {
					// delete after 24 hours
					if _, ok := deleteBuffer[jobState.Name]; !ok {
						deleteBuffer[jobState.Name] = time.Now().Add(time.Hour * 24)
						log.SLogger.Infow("queued for delete job", "jobname", jobState.Name)
					}
				} else if jobState.JobStatus == state.JobStatusSucceed {
					// just delete the mpijob , keep the uploader job and pvc
					// when status is succeed , either the job is succeeded or the job is time up
					job.DeleteJob(jobState.Name, job.JobTypeMPI, false)
				} else { // triggered by user , delete all related resources(uploader job and pvcs)
					job.DeleteJob(jobState.Name, job.JobTypeMPI, true)
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
		for jobName, deleteTime := range deleteBuffer {
			if deleteTime.Before(time.Now()) {
				job.DeleteJob(jobName, job.JobTypeMPI, true)
				delete(deleteBuffer, jobName)
			}
		}
	}
}

func startUploadInfo(done chan struct{}) {
	ch := make(chan communication.UploadPayload, 1)
	// collect data
	go func() {
		for {
			js, err := job.GetJobStates()
			if err != nil {
				log.SLogger.Errorw("get job state error", "error", err)
			}

			select {
			case ch <- communication.UploadPayload{
				CPodID:       config.CPOD_ID,
				ResourceInfo: resource.GetResourceInfo(config.CPOD_ID, "v0.1"), // TODO: 获取版本信息
				JobStatus:    js,
				UpdateTime:   time.Now(),
			}:
				log.SLogger.Infow("refresh upload payload")
			case <-done:
				break
			}
			time.Sleep(time.Second * 10)
		}
	}()

	// upload data , even data is not updated
	go func() {
		var payload communication.UploadPayload
		for {
			select {
			case payload = <-ch:
			case <-done:
				break
			default:
				// do nothing ,  still do the upload but data will be old
				log.SLogger.Warn("cpod status data is not refreshed , upload the old data")
			}
			b := communication.UploadCPodStatus(payload)
			if !b {
				log.SLogger.Warn("upload cpod status data failed")
			} else {
				log.SLogger.Info("uploaded cpod status data")
			}
			time.Sleep(time.Second * 10)
		}
	}()
}

// clean up completed uploader job and pvc when job is deleted
func startCleanUp(done chan struct{}, ns string) {
	go func() {
		//wait to delete
		deleteBuffer := map[string]time.Time{}
		for {
			// find k8s jobs whose state is complete
			completeJobs := []string{}
			k8sJobs, err := clientgo.GetK8SJobs(ns)
			if err != nil {
				log.SLogger.Errorw("error when get k8s job", "namespace", ns, "error", err)
			}
			for _, k8sJob := range k8sJobs.Items {
				// check k8sJob is complete
				complete := false
				for _, cond := range k8sJob.Status.Conditions {
					if cond.Type == batchv1.JobComplete && cond.Status == corev1.ConditionTrue {
						complete = true
						break
					}
				}
				if complete {
					// if not in delete buffer
					if _, ok := deleteBuffer[k8sJob.Name]; !ok {
						completeJobs = append(completeJobs, k8sJob.Name)
					}
				}
			}
			if len(completeJobs) != 0 {
				// get all mpijobs
				mpiJobStates, err := job.GetJobStates()
				if err != nil {
					log.SLogger.Errorw("get jobs state error", "error", err)
					continue
				}
				MPIjobs := map[string]struct{}{}
				for _, j := range mpiJobStates {
					MPIjobs[j.Name] = struct{}{}
				}
				// delete k8s job and related pvc
				for _, completeJob := range completeJobs {
					// when job is deleted
					if _, ok := MPIjobs[completeJob]; !ok {
						// delete after 5 minutes
						deleteBuffer[completeJob] = time.Now().Add(time.Minute * 5)
						log.SLogger.Infow("queued for delete job related resources", "jobname", completeJob)
					}
				}
			}

			for jobName, deleteTime := range deleteBuffer {
				if deleteTime.Before(time.Now()) {
					job.DeleteJobRelated(jobName)
					delete(deleteBuffer, jobName)
				}
			}

			select {
			case <-done:
				break
			default:
				time.Sleep(time.Minute)
			}
		}
	}()
}
