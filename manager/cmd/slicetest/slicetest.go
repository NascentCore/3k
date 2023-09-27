package main

// NO_TEST_NEEDED
// 针对一些功能点的测试
import (
	"encoding/json"
	"io/fs"
	"os"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/job"
)

func main() {
	createMPIJob()
	deleteMPIJob()
	getNodeInfo()
}

func createMPIJob() {
	t := job.Job{
		JobID:                "mpijobtest1111",
		Namespace:            "cpq",
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
		Namespace:            "cpq",
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
	os.WriteFile("nodes.json", b, fs.ModeAppend)
}
