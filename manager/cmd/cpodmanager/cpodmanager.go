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
		jobs := communication.GetJobs(cpodid)
		for _, job := range jobs {
			err := job.Run()
			if err != nil {
				// TODO:  do something else
				log.SLogger.Errorw("Job run failed",
					"job", job,
					"error", err)
			} else {
				log.SLogger.Infow("Job created",
					"job", job)
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
			select {
			case ch <- communication.UploadPayload{
				CPodID:       cpodid,
				ResourceInfo: resource.GetResourceInfo(cpodid, "v0.1"), // TODO: 获取版本信息
				JobStatus:    job.GetJobState(),
				UpdateTime:   time.Now(),
			}:
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
		}
	}()
}
