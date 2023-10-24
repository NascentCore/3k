package job

import (
	"fmt"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	modeluploader "sxwl/3k/manager/pkg/model-uploader"
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
		err := kubeflowmpijob.MPIJob{
			Name:                 j.JobID,
			Namespace:            JobNamespace,
			Image:                j.Image,
			DataPath:             j.DataPath,
			CKPTPath:             j.CKPTPath,
			PretrainModelPath:    "",
			ModelSavePath:        j.ModelPath,
			GPURequiredPerWorker: j.GPURequiredPerWorker,
			Replicas:             j.Replicas,
		}.Run()
		if err != nil {
			return err
		}
		//同时启动Upload Job
		return clientgo.ApplyWithJsonData("cpod", "batch", "v1", "jobs",
			modeluploader.GenK8SJobJsonData(j.JobID, "", "", "/data", []interface{}{"", ""}))
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
			PretrainModelPath:    "",
			ModelSavePath:        j.ModelPath,
			GPURequiredPerWorker: j.GPURequiredPerWorker,
			Replicas:             j.Replicas,
		}.Delete()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}
