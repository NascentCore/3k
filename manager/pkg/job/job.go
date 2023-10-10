package job

import (
	"fmt"
	commonerrors "sxwl/3k/common/errors"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
)

// NO_TEST_NEEDED

type (
	Stage int
	Type  int
)

const (
	JobStageStart     Stage = 0
	JobStageRunning   Stage = 1
	JobStagePending   Stage = 2
	JobStageFinished  Stage = 3
	JobStageUploading Stage = 4
)

const (
	JobTypeMPI Type = 0
)

type JobStatus struct {
	JobID    string
	JobStage Stage
}

type Job struct {
	JobID                string
	Namespace            string
	JobType              Type
	Image                string
	DataPath             string
	CKPTPath             string
	GPURequiredPerWorker int
	Replicas             int
}

func (j Job) Run() error {
	if j.JobType == JobTypeMPI {
		return kubeflowmpijob.MPIJob{
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
		return kubeflowmpijob.MPIJob{
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

func GetJobStatus() []JobStatus {
	return nil
}
