package job

import (
	"errors"
	kubeflowmpijob "sxwl/3k/manager/pkg/kubeflow-mpijob"
)

// NO_TEST_NEEDED

type JobStage int
type JobType int

const (
	JobStageStart     JobStage = 0
	JobStageRunning   JobStage = 1
	JobStagePending   JobStage = 2
	JobStageFinished  JobStage = 3
	JobStageUploading JobStage = 4
)

const (
	JobTypeMPI JobType = 0
)

type JobStatus struct {
	JobID string
	Stage JobStage
}

type Job struct {
	JobID       string
	Namespace   string
	JobType     JobType
	Image       string
	DataPath    string
	CKPTPath    string
	GPURequired int
	Replicas    int
}

func (t Job) Run() error {
	if t.JobType == JobTypeMPI {
		return kubeflowmpijob.KubeFlowMPIJob{
			Name:        t.JobID,
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
