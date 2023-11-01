package main

// NO_TEST_NEEDED

import (
	"os"
	"sxwl/3k/manager/pkg/auth"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/communication"
	"sxwl/3k/manager/pkg/job"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/resource"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
)

func main() {
	if len(os.Args) != 4 {
		log.Logger.Error("Usage : cpodmanager  [ USER_ID ] [ CPOD_ID ] [ BASE_URL ]")
		os.Exit(1)
	}

	// check parameters
	userid := os.Args[1]
	if err := auth.CheckUserId(userid); err != nil {
		panic(err)
	}
	cpodid := os.Args[2]
	if err := auth.CheckCPodId(cpodid); err != nil {
		panic(err)
	}

	base_url := os.Args[3]
	if err := communication.CheckURL(base_url); err != nil {
		panic(err)
	}
	communication.SetBaseURL(base_url)
	clientgo.InitClient()

	// upload resource & task info , also as heartbeat , indicate this cpod is alive and ready for tasks
	var done chan struct{}
	startUploadInfo(done, cpodid)
	// clean mpijob job and pvc in specified namespace
	startCleanUp(done, "cpod")
	// get tasks , then run them !!!
	for {
		//jobSet A
		jobs, err := communication.GetJobs(cpodid)
		if err != nil {
			log.SLogger.Errorw("get jobs from portal error", "error", err)
			continue
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
		//delete all jobs in B whose state is not NoMoreChange and not in A
		for _, jobState := range jobStates {
			if !jobState.JobStatus.NoMoreChange() {
				if _, ok := jobSetA[jobState.Name]; !ok {
					//if delete failed , just skip
					err := job.Job{
						JobID:   jobState.Name,
						JobType: "MPI",
					}.Stop()
					if err != nil {
						log.SLogger.Errorw("Job delete failed",
							"job name", jobState.Name)
					} else {
						log.SLogger.Infow("Job deleted",
							"job", jobState.Name)
					}
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
		time.Sleep(time.Second * 10)
	}
}

func startUploadInfo(done chan struct{}, cpodid string) {
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
				CPodID:       cpodid,
				ResourceInfo: resource.GetResourceInfo(cpodid, "v0.1"), // TODO: 获取版本信息
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

// just clean up the succeed jobs
// keep the failed jobs , convenient for problem detection
func startCleanUp(done chan struct{}, ns string) {
	go func() {
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
					// check whether corresponding mpijob is succeed
					completeJobs = append(completeJobs, k8sJob.Name)
				}
			}
			// get all succeeded mpijobs
			mpiJobStates, err := job.GetJobStates()
			if err != nil {
				log.SLogger.Errorw("get jobs state error", "error", err)
				continue
			}
			succeededMPIjobs := map[string]struct{}{}
			for _, j := range mpiJobStates {
				if j.JobStatus == state.JobStatusSucceed {
					succeededMPIjobs[j.Name] = struct{}{}
				}
			}
			// delete k8s job \ mpijob and related pvc in the both jobset above
			for _, completeJob := range completeJobs {
				if _, ok := succeededMPIjobs[completeJob]; ok {
					//delete , error occurs , just log
					err = clientgo.DeleteK8SJob(ns, completeJob)
					if err != nil {
						log.SLogger.Errorw("delete k8s job failed", "error", err)
					}
					err = clientgo.DeletePVC(ns, kubeflowmpijob.GetCKPTPVCName(completeJob))
					if err != nil {
						log.SLogger.Errorw("delete ckpt pvc failed", "error", err)
					}
					err = clientgo.DeletePVC(ns, kubeflowmpijob.GetModelSavePVCName(completeJob))
					if err != nil {
						log.SLogger.Errorw("delete modelsave pvc failed", "error", err)
					}
					err = job.Job{
						JobID:   completeJob,
						JobType: job.JobTypeMPI,
					}.Stop()
					if err != nil {
						log.SLogger.Errorw("delete mpi job failed", "error", err)
					}
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
