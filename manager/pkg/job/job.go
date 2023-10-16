package job

import (
	"fmt"
	commonerrors "sxwl/3k/common/errors"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
)

// NO_TEST_NEEDED

type (
	Type int
)

const (
	JobTypeMPI   Type = 0
	JobNamespace      = "cpod"
)

type Job struct {
	JobID                string
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
			Namespace:            JobNamespace, //all job runs in cpod namespace
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
			Namespace:            JobNamespace,
			Image:                j.Image,
			DataPath:             j.DataPath,
			CKPTPath:             j.CKPTPath,
			GPURequiredPerWorker: j.GPURequiredPerWorker,
			Replicas:             j.Replicas,
		}.Delete()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %d", j.JobType))
}
