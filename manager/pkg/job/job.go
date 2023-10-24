package job

import (
	"fmt"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	commonerrors "sxwl/3k/pkg/utils/errors"
)

// NO_TEST_NEEDED

type (
	Type string
)

const (
	JobTypeMPI   Type = "MPI" //与云市场一致
	JobNamespace      = "cpod"
)

type Job struct {
	JobID                string
	JobType              Type
	Image                string
	DataPath             string
	CKPTPath             string
	ModelPath            string
	GPUType              string
	GPURequiredPerWorker int
	Replicas             int
	HuggingFaceURL       string
	Duration             int
	StopType             int
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
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
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
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}
