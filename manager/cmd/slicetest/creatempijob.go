package main

import (
	"sxwl/3k/manager/pkg/task"
)

func main() {
	t := task.Task{
		TaskID:      "mpijobtest1111",
		Namespace:   "cpq",
		TaskType:    task.TaskTypeMPI,
		Image:       "swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim",
		DataPath:    "",
		CKPTPath:    "./ds_experiments",
		GPURequired: 8,
		Replicas:    4,
	}
	err := t.Run()
	if err != nil {
		panic(err)
	}
}
