package main

// TODO(peiqing): Turn this file into a actual go test file.

// NO_TEST_NEEDED

import (
	"encoding/json"
	"io/fs"
	"os"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/job"
)

func main() {
	clientgo.InitClient()
	createMPIJob()
	deleteMPIJob()
	getNodeInfo()
}

func createMPIJob() {
	t := job.Job{
		JobID:                "mpijobtest1111",
		JobType:              job.JobTypeMPI,
		Image:                "swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim",
		DataPath:             "",
		CKPTPath:             "./ds_experiments",
		GPURequiredPerWorker: 8,
		Replicas:             4,
	}
	err := t.Run()
	if err != nil {
		panic(err)
	}
}

func deleteMPIJob() {
	t := job.Job{
		JobID:                "mpijobtest1111",
		JobType:              job.JobTypeMPI,
		Image:                "swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim",
		DataPath:             "",
		CKPTPath:             "./ds_experiments",
		GPURequiredPerWorker: 8,
		Replicas:             4,
	}
	err := t.Stop()
	if err != nil {
		panic(err)
	}
}

func getNodeInfo() {
	nodeinfo, err := clientgo.GetNodeInfo()
	if err != nil {
		panic(err)
	}
	b, err := json.Marshal(nodeinfo)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile("nodes.json", b, fs.ModeAppend); err != nil {
		panic(err)
	}
}
