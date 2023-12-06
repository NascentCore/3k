package main

// NO_TEST_NEEDED

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/communication"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job"
	"sxwl/3k/pkg/job/utils"
	"sxwl/3k/pkg/log"
	portalsync "sxwl/3k/pkg/portal-sync"
	"sxwl/3k/pkg/resource"
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

	portalsync.StartPortalSync()
}

func startUploadInfo(done chan struct{}) {
	ch := make(chan communication.UploadPayload, 1)
	// collect data
	go func() {
		for {
			js, err := job.GetJobStatesWithUploaderInfo()
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
			// find modeluploader jobs whose state is complete
			completeJobs := []string{}
			k8sJobs, err := clientgo.GetK8SJobs(ns)
			if err != nil {
				log.SLogger.Errorw("error when get k8s job", "namespace", ns, "error", err)
			}
			for _, k8sJob := range k8sJobs.Items {
				// parseJobName
				jobName := utils.ParseJobNameFromModelUploader(k8sJob.Name)
				if jobName == "" {
					continue
				}
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
					if _, ok := deleteBuffer[jobName]; !ok {
						completeJobs = append(completeJobs, jobName)
					}
				}
			}
			if len(completeJobs) != 0 {
				// get all jobs
				jobStates, err := job.GetJobStates()
				if err != nil {
					log.SLogger.Errorw("get jobs state error", "error", err)
					continue
				}
				jobs := map[string]struct{}{}
				for _, j := range jobStates {
					jobs[j.Name] = struct{}{}
				}
				// delete k8s job and related pvc
				for _, completeJob := range completeJobs {
					// when job is deleted
					if _, ok := jobs[completeJob]; !ok {
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
