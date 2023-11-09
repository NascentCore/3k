package job

import (
	"fmt"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/config"
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	"sxwl/3k/manager/pkg/log"
	modeluploader "sxwl/3k/manager/pkg/model-uploader"
	commonerrors "sxwl/3k/pkg/utils/errors"
	"time"
)

// NO_TEST_NEEDED

type (
	Type string
)

const (
	JobTypeMPI Type = config.PORTAL_JOBTYPE_MPI
)

type Job struct {
	JobID                string
	JobType              Type
	Image                string
	DataPath             string
	CKPTPath             string
	CKPTVolumeSize       int
	ModelPath            string
	ModelVolumeSize      int
	GPUType              string
	GPURequiredPerWorker int
	Replicas             int
	HuggingFaceURL       string
	Duration             int //单位 分钟
	StopType             int //0 自然终止  1 设定时长
}

func (j Job) Run() error {
	if j.JobType == JobTypeMPI {
		//create pvc for ckpt and modelsave
		//TODO: use the volume input from UI
		err := clientgo.CreatePVC(kubeflowmpijob.GetCKPTPVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.CKPTVolumeSize)
		if err != nil {
			return err
		}
		err = clientgo.CreatePVC(kubeflowmpijob.GetModelSavePVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.ModelVolumeSize)
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
			modeluploader.GenK8SJobJsonData(j.JobID, config.MODELUPLOADER_IMAGE, kubeflowmpijob.GetModelSavePVCName(j.JobID), config.MODELUPLOADER_PVC_MOUNT_PATH))
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}

func (j Job) Stop() error {
	if j.JobType == JobTypeMPI {
		return kubeflowmpijob.MPIJob{
			Name:                 j.JobID,
			Namespace:            config.CPOD_NAMESPACE,
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

func DeleteJob(jobName string, jobType Type, deleteRelated bool) {
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
	err = clientgo.DeletePVC(config.CPOD_NAMESPACE, kubeflowmpijob.GetCKPTPVCName(jobName))
	if err != nil {
		log.SLogger.Errorw("PVC for checkpoint delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("PVC for checkpoint deleted",
			"job", jobName)
	}
	err = clientgo.DeletePVC(config.CPOD_NAMESPACE, kubeflowmpijob.GetModelSavePVCName(jobName))
	if err != nil {
		log.SLogger.Errorw("PVC for modelsave delete failed",
			"job name", jobName)
	} else {
		log.SLogger.Infow("PVC for modelsave deleted",
			"job", jobName)
	}
}
