package job

import (
	"fmt"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
	kubeflowmpijob "sxwl/3k/pkg/job/kubeflow-mpijob"
	kubeflowpytorchjob "sxwl/3k/pkg/job/kubeflow-pytorchjob"
	"sxwl/3k/pkg/job/utils"
	"sxwl/3k/pkg/log"
	modeluploader "sxwl/3k/pkg/model-uploader"
	commonerrors "sxwl/3k/pkg/utils/errors"
	"time"
)

// NO_TEST_NEEDED

type (
	JobType  string
	StopType int
)

const (
	JobTypeMPI           JobType  = config.PORTAL_JOBTYPE_MPI
	JobTypePytorch       JobType  = config.PORTAL_JOBTYPE_PYTORCH
	StopTypeWithLimit    StopType = config.PORTAL_STOPTYPE_WITHLIMIT
	StopTypeWithoutLimit StopType = config.PORTAL_STOPTYPE_VOLUNTARY_STOP
)

type Job struct {
	JobID                string
	JobType              JobType
	Image                string
	DataPath             string
	DataUrl              string
	CKPTPath             string
	PretrainModelPath    string
	PretrainModelUrl     string
	CKPTVolumeSize       int
	ModelPath            string
	ModelVolumeSize      int
	GPUType              string
	GPURequiredPerWorker int
	Replicas             int
	HuggingFaceURL       string
	Duration             int      //单位 分钟
	StopType             StopType //0 自然终止  1 设定时长
}

func (j Job) createPVCs() error {
	err := clientgo.CreatePVCIFNotExist(utils.GetCKPTPVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.CKPTVolumeSize)
	if err != nil {
		return err
	}
	err = clientgo.CreatePVCIFNotExist(utils.GetModelSavePVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.ModelVolumeSize)
	if err != nil {
		return err
	}
	return nil
}

func (j Job) runMPIJob() error {
	err := j.createPVCs()
	if err != nil {
		return err
	}
	dl := time.Now().Add(time.Duration(time.Hour * 24 * 365 * 50)) // super long time
	if j.StopType == 1 {
		dl = time.Now().Add(time.Minute * time.Duration(j.Duration))
	}
	err = kubeflowmpijob.MPIJob{
		Name:                 j.JobID,
		Namespace:            config.CPOD_NAMESPACE,
		Image:                j.Image,
		DataPath:             j.DataPath,
		CKPTPath:             j.CKPTPath,
		PretrainModelPath:    "",
		ModelSavePath:        j.ModelPath,
		GPUType:              j.GPUType,
		GPURequiredPerWorker: j.GPURequiredPerWorker,
		Replicas:             j.Replicas,
		Deadline:             dl.Format(config.TIME_FORMAT_FOR_K8S_LABEL),
	}.Run()
	if err != nil {
		return err
	}
	//同时启动Upload Job
	return clientgo.ApplyWithJsonData(config.CPOD_NAMESPACE, "batch", "v1", "jobs",
		modeluploader.GenK8SJobJsonData(j.JobID, config.MODELUPLOADER_IMAGE, utils.GetModelSavePVCName(j.JobID), config.MODELUPLOADER_PVC_MOUNT_PATH))
}

func (j Job) runPytorchJob() error {
	//create pvc for ckpt and modelsave
	dataPVC, err := utils.GetDatasetPVC(j.DataUrl)
	fmt.Println("dataPVC : ", dataPVC, err)
	if err != nil {
		return err
	}
	modelPVC, err := utils.GetModelPVC(j.PretrainModelUrl)
	fmt.Println("modelPVC : ", modelPVC, err)
	if err != nil {
		return err
	}
	j.createPVCs()
	if err != nil {
		return err
	}
	dl := time.Now().Add(time.Duration(time.Hour * 24 * 365 * 50)) // super long time
	if j.StopType == StopTypeWithLimit {
		dl = time.Now().Add(time.Minute * time.Duration(j.Duration))
	}

	err = kubeflowpytorchjob.PytorchJob{
		Name:                 j.JobID,
		Namespace:            config.CPOD_NAMESPACE,
		Image:                j.Image,
		DataPath:             j.DataPath,
		DataPVC:              dataPVC,
		CKPTPath:             j.CKPTPath,
		PretrainModelPath:    j.PretrainModelPath,
		PretrainModelPVC:     modelPVC,
		ModelSavePath:        j.ModelPath,
		GPUType:              j.GPUType,
		GPURequiredPerWorker: j.GPURequiredPerWorker,
		Replicas:             j.Replicas,
		Deadline:             dl.Format(config.TIME_FORMAT_FOR_K8S_LABEL),
	}.Run()
	if err != nil {
		return err
	}
	//同时启动Upload Job
	return clientgo.ApplyWithJsonData(config.CPOD_NAMESPACE, "batch", "v1", "jobs",
		modeluploader.GenK8SJobJsonData(j.JobID, config.MODELUPLOADER_IMAGE, utils.GetModelSavePVCName(j.JobID), config.MODELUPLOADER_PVC_MOUNT_PATH))

}

func (j Job) Run() error {
	if j.JobType == JobTypeMPI {
		return j.runMPIJob()
	} else if j.JobType == JobTypePytorch {
		return j.runPytorchJob()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}

func (j Job) Stop() error {
	if j.JobType == JobTypeMPI {
		return kubeflowmpijob.MPIJob{
			Name:      j.JobID,
			Namespace: config.CPOD_NAMESPACE,
		}.Delete()
	} else if j.JobType == JobTypePytorch {
		return kubeflowpytorchjob.PytorchJob{
			Name:      j.JobID,
			Namespace: config.CPOD_NAMESPACE,
		}.Delete()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}

// TODO: suport pytorchjob
func DeleteJob(jobName string, jobType JobType, deleteRelated bool) {
	//delete job first
	err := Job{
		JobID:   jobName,
		JobType: jobType,
	}.Stop()
	if err != nil {
		log.SLogger.Errorw("Job delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("Job deleted",
			"job", jobName)
	}
	// dont care job is deleted or not
	if deleteRelated {
		DeleteJobRelated(jobName)
	}
}

// TODO: suport pytorchjob
func DeleteJobRelated(jobName string) {
	err := clientgo.DeleteK8SJob(config.CPOD_NAMESPACE, jobName)
	if err != nil {
		log.SLogger.Errorw("Uploader Job delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("Uploader Job deleted",
			"job", jobName)
	}
	// dont care uploader job is deleted or not
	err = clientgo.DeletePVC(config.CPOD_NAMESPACE, utils.GetCKPTPVCName(jobName))
	if err != nil {
		log.SLogger.Errorw("PVC for checkpoint delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("PVC for checkpoint deleted",
			"job", jobName)
	}
	err = clientgo.DeletePVC(config.CPOD_NAMESPACE, utils.GetModelSavePVCName(jobName))
	if err != nil {
		log.SLogger.Errorw("PVC for modelsave delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("PVC for modelsave deleted",
			"job", jobName)
	}
}
