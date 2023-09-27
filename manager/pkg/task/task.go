package task

import (
	"errors"
	kubeflowmpijob "sxwl/3k/manager/pkg/kubeflow-mpijob"
)

// NO_TEST_NEEDED

type TaskStage int
type TaskType int

const (
	TaskStageStart     TaskStage = 0
	TaskStageRunning   TaskStage = 1
	TaskStagePending   TaskStage = 2
	TaskStageFinished  TaskStage = 3
	TaskStageUploading TaskStage = 4
)

const (
	TaskTypeMPI TaskType = 0
)

type TaskStatus struct {
	TaskID string
	Stage  TaskStage
}

type Task struct {
	TaskID      string
	Namespace   string
	TaskType    TaskType
	Image       string
	DataPath    string
	CKPTPath    string
	GPURequired int
	Replicas    int
}

func (t Task) Run() error {
	if t.TaskType == TaskTypeMPI {
		return kubeflowmpijob.KubeFlowMPIJob{
			Name:        t.TaskID,
			Namespace:   t.Namespace,
			Image:       t.Image,
			DataPath:    t.DataPath,
			CKPTPath:    t.CKPTPath,
			GPURequired: t.GPURequired,
			Replicas:    t.Replicas,
		}.Run()
	}
	return errors.New("not Impl")
}
