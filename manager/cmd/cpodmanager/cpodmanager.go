package main

// NO_TEST_NEEDED

import (
	"os"
	"sxwl/3k/manager/pkg/auth"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/communication"
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/resource"
	"time"
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
