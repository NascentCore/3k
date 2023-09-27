package job

import (
	"fmt"
	commonerrors "sxwl/3k/common/errors"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
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
	JobID                string
	Namespace            string
	JobType              JobType
	Image                string
	DataPath             string
	CKPTPath             string
	GPURequiredPerWorker int
	Replicas             int
}

func (j Job) Run() error {
	if j.JobType == JobTypeMPI {
		return kubeflowmpijob.KubeFlowMPIJob{
			Name:                 j.JobID,
			Namespace:            j.Namespace,
			Image:                j.Image,
			DataPath:             j.DataPath,
			CKPTPath:             j.CKPTPath,
			GPURequiredPerWorker: j.GPURequiredPerWorker,
			Replicas:             j.Replicas,
		}.Run()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %d", j.JobType))
}

func (j Job) Stop() error {
	if j.JobType == JobTypeMPI {
		return kubeflowmpijob.KubeFlowMPIJob{
			Name:                 j.JobID,
			Namespace:            j.Namespace,
			Image:                j.Image,
			DataPath:             j.DataPath,
			CKPTPath:             j.CKPTPath,
			GPURequiredPerWorker: j.GPURequiredPerWorker,
			Replicas:             j.Replicas,
		}.Delete()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %d", j.JobType))
}
