package communication

import (
	"sxwl/3k/manager/pkg/cluster"
	"sxwl/3k/manager/pkg/task"
	"time"
)

type UploadPayload struct {
	CPodID       string
	TaskStatus   []task.TaskStatus
	ResourceDesc cluster.ResourceDesc
	UpdateTime   time.Time
}

type TaskPayload struct {
	Tasks []task.Task
}

// communicate with market manager get tasks should run in this cpod
func GetTasks(cpodid string) []task.Task {
	//TODO
	return []task.Task{}
}

func UploadCPodStatus(up UploadPayload) bool {
	//TODO
	return true
}
