package job

import (
	"fmt"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
	generaljob "sxwl/3k/pkg/job/general-job"
	kubeflowmpijob "sxwl/3k/pkg/job/kubeflow-mpijob"
	kubeflowpytorchjob "sxwl/3k/pkg/job/kubeflow-pytorchjob"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/job/utils"
	"sxwl/3k/pkg/log"
	commonerrors "sxwl/3k/pkg/utils/errors"
	"time"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
)

// NO_TEST_NEEDED

type (
	StopType int
)

const (
	StopTypeWithLimit    StopType = config.PORTAL_STOPTYPE_WITHLIMIT
	StopTypeWithoutLimit StopType = config.PORTAL_STOPTYPE_VOLUNTARY_STOP
)

type Job struct {
	JobID                string
	JobType              state.JobType
	Image                string
	DataPath             string
	DataName             string
	CKPTPath             string
	PretrainModelPath    string
	PretrainModelName    string
	CKPTVolumeSize       int
	ModelPath            string
	ModelVolumeSize      int
	GPUType              string
	GPURequiredPerWorker int
	Replicas             int
	Command              []string
	Envs                 map[string]string
	Duration             int      //Ã¥ÂÂÃ¤Â½Â Ã¥ÂÂÃ©ÂÂ
	StopType             StopType //0 Ã¨ÂÂªÃ§ÂÂ¶Ã§Â»ÂÃ¦Â­Â¢  1 Ã¨Â®Â¾Ã¥Â®ÂÃ¦ÂÂ¶Ã©ÂÂ¿
}

type Interface interface {
	Run() error
	Delete() error
}

func (j Job) createPVCs() error {
	var err error
	if j.CKPTPath != "" { // if ckpt not specified , just skip
		err = clientgo.CreatePVCIFNotExist(utils.GetCKPTPVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.CKPTVolumeSize)
		if err != nil {
			return err
		}
	}
	// however , modelsave pvc must be there
	err = clientgo.CreatePVCIFNotExist(utils.GetModelSavePVCName(j.JobID), config.CPOD_NAMESPACE, config.STORAGE_CLASS_TO_CREATE_PVC, config.CPOD_CREATED_PVC_ACCESS_MODE, j.ModelVolumeSize)
	if err != nil {
		return err
	}
	return nil
}

func (j Job) toActualJob() (Interface, error) {
	dataPVC := ""
	var err error
	if j.DataName != "" { // if not specified , skip
		dataPVC, err = utils.GetDatasetPVC(j.DataName)
		if err != nil {
			return nil, err
		}
	}
	modelPVC := ""
	if j.PretrainModelName != "" { // if not specified , skip
		modelPVC, err = utils.GetModelPVC(j.PretrainModelName)
		if err != nil {
			return nil, err
		}
	}
	dl := time.Now().Add(time.Duration(time.Hour * 24 * 365 * 50)) // super long time
	if j.StopType == StopTypeWithLimit {
		dl = time.Now().Add(time.Minute * time.Duration(j.Duration))
	}
	if j.JobType == state.JobTypePytorch {
		return kubeflowpytorchjob.PytorchJob{
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
			Command:              j.Command,
			Deadline:             dl.Format(config.TIME_FORMAT_FOR_K8S_LABEL),
		}, nil
	} else if j.JobType == state.JobTypeMPI {
		return kubeflowmpijob.MPIJob{
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
			Command:              j.Command,
			Replicas:             j.Replicas,
			Deadline:             dl.Format(config.TIME_FORMAT_FOR_K8S_LABEL),
		}, nil
	} else if j.JobType == state.JobTypeGeneral {
		return generaljob.GeneralJob{
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
			Command:              j.Command,
			Envs:                 j.Envs,
			Deadline:             dl.Format(config.TIME_FORMAT_FOR_K8S_LABEL),
		}, nil
	} else {
		return nil, commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
	}
}

func (j Job) Run() error {
	aj, err := j.toActualJob()
	if err != nil {
		return err
	}
	j.createPVCs()
	if err != nil {
		return err
	}
	err = aj.Run()
	if err != nil {
		return err
	}
	//Ã¥ÂÂÃ¦ÂÂ¶Ã¥ÂÂ¯Ã¥ÂÂ¨Upload Job
	return clientgo.ApplyWithJsonData(config.CPOD_NAMESPACE, "batch", "v1", "jobs",
		utils.GenK8SJobJsonData(j.JobID, config.MODELUPLOADER_IMAGE, utils.GetModelSavePVCName(j.JobID), config.MODELUPLOADER_PVC_MOUNT_PATH))

}

func (j Job) Stop() error {
	if j.JobType == state.JobTypeMPI {
		return kubeflowmpijob.MPIJob{
			Name:      j.JobID,
			Namespace: config.CPOD_NAMESPACE,
		}.Delete()
	} else if j.JobType == state.JobTypePytorch {
		return kubeflowpytorchjob.PytorchJob{
			Name:      j.JobID,
			Namespace: config.CPOD_NAMESPACE,
		}.Delete()
	} else if j.JobType == state.JobTypeGeneral {
		return generaljob.GeneralJob{
			Name:      j.JobID,
			Namespace: config.CPOD_NAMESPACE,
		}.Delete()
	}
	return commonerrors.UnImpl(fmt.Sprintf("job of type %s", j.JobType))
}

func DeleteJob(jobName string, jobType state.JobType, deleteRelated bool) {
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
	err := clientgo.DeleteK8SJob(config.CPOD_NAMESPACE, utils.GenModelUploaderJobName(jobName))
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

func GetState(namespace, name string) (state.State, error) {
	s, err := kubeflowmpijob.GetState(namespace, name)
	if err != nil {
		if !k8serrors.IsNotFound(err) {
			return state.State{}, err
		} else {
			// else try next job type
		}
	} else {
		return s, nil
	}
	s, err = kubeflowpytorchjob.GetState(namespace, name)
	if err != nil {
		if !k8serrors.IsNotFound(err) {
			return state.State{}, err
		}
	} else {
		return s, nil
	}
	s, err = generaljob.GetState(namespace, name)
	if err != nil {
		return state.State{}, err
	}
	return s, err
}
