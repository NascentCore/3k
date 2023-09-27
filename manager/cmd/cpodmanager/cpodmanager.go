package main

// NO_TEST_NEEDED

import (
	"fmt"
	"os"
	"sxwl/3k/manager/pkg/auth"
	"sxwl/3k/manager/pkg/cluster"
	"sxwl/3k/manager/pkg/cluster/kubectl"
	"sxwl/3k/manager/pkg/communication"
	"time"
)

func main() {
	if len(os.Args) != 3 {
		os.Exit(1)
		fmt.Println("Usage : cpodmanager  [ USER_ID ] [ CPOD_ID ]")
		return
	}
	userid := os.Args[1]
	if ok, msg := auth.CheckUserId(userid); !ok {
		fmt.Println(msg)
	}
	cpodid := os.Args[2]
	if ok, msg := auth.CheckCPodId(cpodid); !ok {
		fmt.Println(msg)
	}

	//upload resource & task info , also as heartbeat , indicate this cpod is alive and ready for tasks
	var done chan struct{}
	startUploadResourceInfo(done, cpodid)
	//get tasks , then run them !!!
	for {
		tasks := communication.GetTasks(cpodid)
		for _, task := range tasks {
			task.Run()
		}
		time.Sleep(time.Second * 10)
	}
}

func startUploadResourceInfo(done chan struct{}, cpodid string) {
	ch := make(chan communication.UploadPayload, 1)
	//collect data
	go func() {
		for {
			select {
			case ch <- communication.UploadPayload{CPodID: cpodid, ResourceDesc: cluster.GetResourceDesc(), TaskStatus: kubectl.GetTaskStatus(), UpdateTime: time.Now()}:
			case <-done:
				break
			}
			time.Sleep(time.Second * 10)
		}
	}()

	//upload data , even data is not updated
	go func() {
		var payload communication.UploadPayload
		for {
			select {
			case payload = <-ch:
			case <-done:
				break
			default:
				//do nothing ,  still do the upload but data will be old
			}
			communication.UploadCPodStatus(payload)
		}
	}()
}
