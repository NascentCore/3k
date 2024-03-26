package synchronizer

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	InferenceJob = "inferencejob"
	TrainningJob = "trainningjob"
)

type jobBuffer struct {
	mtj map[string]sxwl.PortalTrainningJob
	mij map[string]sxwl.PortalInferenceJob
	mu  *sync.RWMutex
}

type SyncJob struct {
	kubeClient       client.Client
	scheduler        sxwl.Scheduler
	createFailedJobs jobBuffer
	preparingJobs    jobBuffer
	logger           logr.Logger
}

func NewSyncJob(kubeClient client.Client, scheduler sxwl.Scheduler, logger logr.Logger) *SyncJob {
	return &SyncJob{
		kubeClient: kubeClient,
		scheduler:  scheduler,
		createFailedJobs: jobBuffer{
			mtj: map[string]sxwl.PortalTrainningJob{},
			mij: map[string]sxwl.PortalInferenceJob{},
			mu:  new(sync.RWMutex),
		},
		preparingJobs: jobBuffer{
			mtj: map[string]sxwl.PortalTrainningJob{},
			mij: map[string]sxwl.PortalInferenceJob{},
			mu:  new(sync.RWMutex),
		},
		logger: logger,
	}
}

func jobTypeCheck(jobtype string) (v1beta1.JobType, bool) {
	if strings.ToLower(jobtype) == string(v1beta1.JobTypeMPI) {
		return v1beta1.JobTypeMPI, true
	}
	if strings.ToLower(jobtype) == string(v1beta1.JobTypePytorch) {
		return v1beta1.JobTypePytorch, true
	}
	return "", false
}

// first retrieve twn job sets , portal job set and cpod job set
// for jobs in portal not in cpod , create it
// for jobs in cpod not in portal , if it's running , delete it
func (s *SyncJob) Start(ctx context.Context) {
	s.logger.Info("sync job")

	portalTrainningJobs, portalInferenceJobs, err := s.scheduler.GetAssignedJobList()
	if err != nil {
		s.logger.Error(err, "failed to list job")
		return
	}
	s.logger.Info("assigned trainning job", "jobs", portalTrainningJobs)
	s.logger.Info("assigned inference job", "jobs", portalInferenceJobs)
	s.processTrainningJobs(ctx, portalTrainningJobs)
	s.processFinetune(ctx, portalTrainningJobs)
	s.processInferenceJobs(ctx, portalInferenceJobs)

}

func (s *SyncJob) processFinetune(ctx context.Context, portaljobs []sxwl.PortalTrainningJob) {
	var finetunes v1beta1.FineTuneList
	err := s.kubeClient.List(ctx, &finetunes, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		s.logger.Error(err, "failed to list finetunejob")
	}
	for _, job := range portaljobs {
		if job.JobType != "Codeless" {
			continue
		}
		s.logger.Info("finetune job", "finetune", job.JobName)
		exists := false
		for _, finetune := range finetunes.Items {
			if finetune.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			//create
			newJob := v1beta1.FineTune{
				ObjectMeta: metav1.ObjectMeta{
					Name:      job.JobName,
					Namespace: v1beta1.CPOD_NAMESPACE,
					Labels: map[string]string{
						v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
						v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
					},
				},
				Spec: v1beta1.FineTuneSpec{
					Model:          job.PretrainModelName,
					DatasetStorage: job.DatasetId,
					Upload:         true,
					HyperParameters: map[string]string{
						"n_epochs":                 job.Epochs,
						"learning_rate_multiplier": job.LearningRate,
						"batch_size":               job.BatchSize,
					},
					GPUCount:   int32(job.GpuNumber),
					GPUProduct: job.GpuType,
				},
			}

			if err = s.kubeClient.Create(ctx, &newJob); err != nil {
				s.logger.Error(err, "failed to create finetune", "job", newJob)
			}
		}
	}

	for _, finetune := range finetunes.Items {
		exists := false
		for _, job := range portaljobs {
			if finetune.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			if err = s.kubeClient.Delete(ctx, &finetune); err != nil {
				s.logger.Error(err, "failed to delete finetune")
				return
			}
		}
	}
}

func (s *SyncJob) processTrainningJobs(ctx context.Context, portaljobs []sxwl.PortalTrainningJob) {
	var cpodTrainningJobs v1beta1.CPodJobList
	err := s.kubeClient.List(ctx, &cpodTrainningJobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		s.logger.Error(err, "failed to list trainningjob")
		return
	}

	for _, job := range portaljobs {
		if job.JobType == "Codeless" {
			continue
		}

		exists := false
		for _, cpodTrainningJob := range cpodTrainningJobs.Items {
			if cpodTrainningJob.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			var cmd []string
			if job.Command != "" {
				cmd = strings.Split(job.Command, " ")
			}
			var envs []v1.EnvVar
			for k, v := range job.Envs {
				envs = append(envs, v1.EnvVar{Name: k, Value: v})
			}
			duration := 0
			if job.StopType == v1beta1.PORTAL_STOPTYPE_WITHLIMIT && job.StopTime > 0 {
				duration = job.StopTime
			}

			if job.PretrainModelId != "" {
				//判断指定的预训练模型是否存在
				exists, done, err := s.checkModelExistence(ctx, v1beta1.CPOD_NAMESPACE, job.PretrainModelId)
				if err != nil {
					s.logger.Error(err, "failed to check model existence", "modelid", job.PretrainModelId)
					continue
				}
				if exists {
					if !done {
						s.logger.Info("Model is preparing.", "jobname", job.JobName, "modelid", job.PretrainModelId)
						continue
					}
				} else { // modelstorage not exist
					s.logger.Info("model not exists , starting downloader task , task will be started when downloader task finish",
						"jobname", job.JobName, "modelid", job.PretrainModelId)
					// return preparing status during the downloader task.
					s.addPreparingTrainningJob(job)
					//create PVC
					ossAK := os.Getenv("AK")
					ossSK := os.Getenv("AS")
					storageClassName := os.Getenv("STORAGECLASS")
					ossPath := ResourceToOSSPath(Model, job.PretrainModelName)
					pvcName := ModelPVCName(ossPath)
					//storageName := ModelCRDName(ossPath)
					modelSize := fmt.Sprintf("%d", job.PretrainModelSize)
					//pvcsize is 1.2 * modelsize
					pvcSize := fmt.Sprintf("%dMi", job.PretrainModelSize*12/10/1024/1024)
					err := s.createPVC(ctx, pvcName, pvcSize, storageClassName)
					if err != nil {
						s.logger.Error(err, "create pvc failed", "jobname", job.JobName, "modelid", job.PretrainModelId)
						continue
					} else {
						s.logger.Info("pvc created", "jobname", job.JobName, "modelid", job.PretrainModelId)
					}
					//create ModelStorage
					err = s.createModelStorage(ctx, job.PretrainModelId, job.PretrainModelName, pvcName)
					if err != nil {
						s.logger.Error(err, "create modelstorage failed", "jobname", job.JobName, "modelid", job.PretrainModelId)
						continue
					} else {
						s.logger.Info("modelstorage created", "jobname", job.JobName, "modelid", job.PretrainModelId)
					}
					//create DownloaderJob
					err = s.createDownloaderJob(ctx, "model", pvcName, ModelDownloadJobName(ossPath), job.PretrainModelId, modelSize, job.PretrainModelUrl, ossAK, ossSK)
					if err != nil {
						s.logger.Error(err, "create downloader job failed", "jobname", job.JobName, "modelid", job.PretrainModelId)
						continue
					} else {
						s.logger.Info("downloader job created", "jobname", job.JobName, "modelid", job.PretrainModelId)
					}
					continue
				}
			}
			if job.DatasetId != "" {
				//判断指定的预训练模型是否存在
				exists, done, err := s.checkDatasetExistence(ctx, v1beta1.CPOD_NAMESPACE, job.DatasetId)
				if err != nil {
					s.logger.Error(err, "failed to check dataset existence", "datasetid", job.DatasetId)
					continue
				}
				if exists {
					if !done {
						s.logger.Info("Dataset is preparing.", "jobname", job.JobName, "datasetid", job.DatasetId)
						continue
					}
				} else { // dataset not exist
					s.logger.Info("dataset not exists , starting downloader task , task will be started when downloader task finish",
						"jobname", job.JobName, "datasetid", job.DatasetId)
					// return preparing status during the downloader task.
					s.addPreparingTrainningJob(job)
					//create PVC
					ossAK := os.Getenv("AK")
					ossSK := os.Getenv("AS")
					storageClassName := os.Getenv("STORAGECLASS")
					ossPath := ResourceToOSSPath(Dataset, job.DatasetName)
					pvcName := DatasetPVCName(ossPath)
					datasetSize := fmt.Sprintf("%d", job.DatasetSize)
					//pvcsize is 1.2 * datasetsize
					pvcSize := fmt.Sprintf("%dMi", job.DatasetSize*12/10/1024/1024)
					err := s.createPVC(ctx, pvcName, pvcSize, storageClassName)
					if err != nil {
						s.logger.Error(err, "create pvc failed", "jobname", job.JobName, "datasetid", job.DatasetId)
						continue
					} else {
						s.logger.Info("pvc created", "jobname", job.JobName, "datasetid", job.DatasetId)
					}
					//create DatasetStorage
					err = s.createDatasetStorage(ctx, job.DatasetId, job.DatasetName, pvcName)
					if err != nil {
						s.logger.Error(err, "create datasetstorage failed", "jobname", job.JobName, "datasetid", job.DatasetId)
						continue
					} else {
						s.logger.Info("datasetstorage created", "jobname", job.JobName, "datasetid", job.DatasetId)
					}
					//create DownloaderJob
					err = s.createDownloaderJob(ctx, "dataset", pvcName, DatasetDownloadJobName(ossPath), job.DatasetId, datasetSize, job.DatasetUrl, ossAK, ossSK)
					if err != nil {
						s.logger.Error(err, "create downloader job failed", "jobname", job.JobName, "datasetid", job.DatasetId)
						continue
					} else {
						s.logger.Info("downloader job created", "jobname", job.JobName, "datasetid", job.DatasetId)
					}
					continue
				}
			}
			var gpuPerWorker int32 = 8
			var replicas int32 = 1
			if job.GpuNumber < 8 {
				gpuPerWorker = int32(job.GpuNumber)
			} else {
				replicas = int32(job.GpuNumber) / 8
			}
			jobType, ok := jobTypeCheck(job.JobType)
			if !ok {
				s.logger.Info("invalid jobtype", "jobtype", job.JobType, "jobname", job.JobName)
				s.addCreateFailedTrainningJob(job)
				continue
			}
			var backoffLimit int32 = int32(job.BackoffLimit)

			newJob := v1beta1.CPodJob{
				ObjectMeta: metav1.ObjectMeta{
					// TODO: create namespace for different tenant
					Namespace: v1beta1.CPOD_NAMESPACE,
					Name:      job.JobName,
					Labels: map[string]string{
						v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
						v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
					},
				},

				Spec: v1beta1.CPodJobSpec{
					JobType:               jobType,
					GPURequiredPerReplica: gpuPerWorker,
					GPUType:               job.GpuType,
					DatasetPath:           job.DatasetPath,
					DatasetName:           job.DatasetId,
					PretrainModelPath:     job.PretrainModelPath,
					PretrainModelName:     job.PretrainModelId,
					CKPTPath:              job.CkptPath,
					CKPTVolumeSize:        int32(job.CkptVol),
					ModelSavePath:         job.ModelPath,
					ModelSaveVolumeSize:   int32(job.ModelVol),
					UploadModel:           true,
					Duration:              int32(duration),
					Image:                 job.ImagePath,
					Command:               cmd,
					Envs:                  envs,
					WorkerReplicas:        replicas,
					BackoffLimit:          &backoffLimit,
				},
			}
			if err = s.kubeClient.Create(ctx, &newJob); err != nil {
				s.addCreateFailedTrainningJob(job)
				s.logger.Error(err, "failed to create trainningjob", "job", newJob)
			} else {
				s.deleteCreateFailedTrainningJob(job.JobName)
				s.deletePreparingTrainningJob(job.JobName)
				s.logger.Info("trainningjob created", "job", newJob)
			}
		}
	}

	for _, cpodTrainningJob := range cpodTrainningJobs.Items {
		// do nothing if job has reached a no more change status
		status, _ := parseStatus(cpodTrainningJob.Status)
		if status == v1beta1.JobFailed || status == v1beta1.JobModelUploaded ||
			status == v1beta1.JobSucceeded || status == v1beta1.JobModelUploading {
			continue
		}
		exists := false
		for _, job := range portaljobs {
			if cpodTrainningJob.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			if err = s.kubeClient.Delete(ctx, &cpodTrainningJob); err != nil {
				s.logger.Error(err, "failed to delete trainningjob")
				return
			}
			s.logger.Info("trainningjob deleted", "jobid", cpodTrainningJob.Name)
		}
	}
}

func (s *SyncJob) processInferenceJobs(ctx context.Context, portaljobs []sxwl.PortalInferenceJob) {
	var cpodInferenceJobs v1beta1.InferenceList
	err := s.kubeClient.List(ctx, &cpodInferenceJobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		s.logger.Error(err, "failed to list inference job")
		return
	}

	for _, job := range portaljobs {
		exists := false
		for _, cpodInferenceJob := range cpodInferenceJobs.Items {
			if cpodInferenceJob.Name == job.ServiceName {
				exists = true
			}
		}
		if !exists {
			template := job.Template
			if template == "" {
				template = "default"
			}
			//create
			newJob := v1beta1.Inference{
				ObjectMeta: metav1.ObjectMeta{
					// TODO: create namespace for different tenant
					Namespace: v1beta1.CPOD_NAMESPACE,
					Name:      job.ServiceName,
					Labels: map[string]string{
						v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
						v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
					},
				},
				// TODO: fill PredictorSpec with infomation provided by portaljob
				Spec: v1beta1.InferenceSpec{
					Predictor: kservev1beta1.PredictorSpec{
						PodSpec: kservev1beta1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "kserve-container",
									Image: "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:v1",
									Command: []string{
										"python",
										"src/api_demo.py",
										"--model_name_or_path",
										"/mnt/models",
										"--template",
										template,
									},
									Env: []v1.EnvVar{
										{
											Name:  "STORAGE_URI",
											Value: "modelstorage://" + job.ModelId,
										},
										{
											Name:  "API_PORT",
											Value: "8080",
										},
									},
									Resources: v1.ResourceRequirements{
										Limits: map[v1.ResourceName]resource.Quantity{
											v1.ResourceCPU:    resource.MustParse("4"),
											v1.ResourceMemory: resource.MustParse("50Gi"),
											"nvidia.com/gpu":  resource.MustParse("1"),
										},
										Requests: map[v1.ResourceName]resource.Quantity{
											v1.ResourceCPU:    resource.MustParse("4"),
											v1.ResourceMemory: resource.MustParse("50Gi"),
											"nvidia.com/gpu":  resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
			}
			if err = s.kubeClient.Create(ctx, &newJob); err != nil {
				s.addCreateFailedInferenceJob(job)
				s.logger.Error(err, "failed to create inference job", "job", newJob)
			} else {
				s.deleteCreateFailedInferenceJob(job.ServiceName)
				s.logger.Info("inference job created", "job", newJob)
			}
		}
	}

	for _, cpodInferenceJob := range cpodInferenceJobs.Items {
		exists := false
		for _, job := range portaljobs {
			if cpodInferenceJob.Name == job.ServiceName {
				exists = true
			}
		}
		if !exists {
			if err = s.kubeClient.Delete(ctx, &cpodInferenceJob); err != nil {
				s.logger.Error(err, "failed to delete inference job")
				return
			}
			s.logger.Info("inference job deleted", "jobid", cpodInferenceJob.Name)
		}
	}
}

func (s *SyncJob) getCreateFailedTrainningJobs() []sxwl.PortalTrainningJob {
	res := []sxwl.PortalTrainningJob{}
	s.createFailedJobs.mu.RLock()
	defer s.createFailedJobs.mu.RUnlock()
	for _, v := range s.createFailedJobs.mtj {
		res = append(res, v)
	}
	return res
}

func (s *SyncJob) addCreateFailedTrainningJob(j sxwl.PortalTrainningJob) {
	if _, ok := s.createFailedJobs.mtj[j.JobName]; ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	s.createFailedJobs.mtj[j.JobName] = j
}

// 如果任务创建成功了，将其从失败任务列表中删除
func (s *SyncJob) deleteCreateFailedTrainningJob(j string) {
	if _, ok := s.createFailedJobs.mtj[j]; !ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	delete(s.createFailedJobs.mtj, j)
}

func (s *SyncJob) getPreparingTrainningJobs() []sxwl.PortalTrainningJob {
	res := []sxwl.PortalTrainningJob{}
	s.preparingJobs.mu.RLock()
	defer s.preparingJobs.mu.RUnlock()
	for _, v := range s.preparingJobs.mtj {
		res = append(res, v)
	}
	return res
}

func (s *SyncJob) addPreparingTrainningJob(j sxwl.PortalTrainningJob) {
	if _, ok := s.preparingJobs.mtj[j.JobName]; ok {
		return
	}
	s.preparingJobs.mu.Lock()
	defer s.preparingJobs.mu.Unlock()
	s.preparingJobs.mtj[j.JobName] = j
}

func (s *SyncJob) deletePreparingTrainningJob(j string) {
	if _, ok := s.preparingJobs.mtj[j]; !ok {
		return
	}
	s.preparingJobs.mu.Lock()
	defer s.preparingJobs.mu.Unlock()
	delete(s.preparingJobs.mtj, j)
}

func (s *SyncJob) getCreateFailedInferenceJobs() []sxwl.PortalInferenceJob {
	res := []sxwl.PortalInferenceJob{}
	s.createFailedJobs.mu.RLock()
	defer s.createFailedJobs.mu.RUnlock()
	for _, v := range s.createFailedJobs.mij {
		res = append(res, v)
	}
	return res
}

func (s *SyncJob) addCreateFailedInferenceJob(j sxwl.PortalInferenceJob) {
	if _, ok := s.createFailedJobs.mij[j.ServiceName]; ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	s.createFailedJobs.mij[j.ServiceName] = j
}

// 如果任务创建成功了，将其从失败任务列表中删除
func (s *SyncJob) deleteCreateFailedInferenceJob(j string) {
	if _, ok := s.createFailedJobs.mij[j]; !ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	delete(s.createFailedJobs.mij, j)
}

// 检查模型是否存在
func (s *SyncJob) checkModelExistence(ctx context.Context, namespace, m string) (bool, bool, error) {
	var ms cpodv1.ModelStorage
	err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: m}, &ms)
	if err != nil {
		if kerrors.IsNotFound(err) {
			return false, false, nil
		}
		return false, false, err
	}
	return true, ms.Status.Phase == "done", nil
}

// 检查数据集是否存在
func (s *SyncJob) checkDatasetExistence(ctx context.Context, namespace, d string) (bool, bool, error) {
	var ds cpodv1.DataSetStorage
	err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: d}, &ds)
	if err != nil {
		if kerrors.IsNotFound(err) {
			return false, false, nil
		}
		return false, false, err
	}
	return true, ds.Status.Phase == "done", nil
}

func (s *SyncJob) createModelStorage(ctx context.Context, modelStorageName, modelName, pvcName string) error {
	m := cpodv1.ModelStorage{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: v1beta1.CPOD_NAMESPACE,
			Name:      modelStorageName,
		},
		Spec: cpodv1.ModelStorageSpec{
			ModelType:             "oss",
			ModelName:             modelName,
			PVC:                   pvcName,
			ConvertTensorRTEngine: false,
		},
	}
	err := s.kubeClient.Create(ctx, &m)
	return err
}

func (s *SyncJob) createDatasetStorage(ctx context.Context, datasetStorageName, datasetName, pvcName string) error {
	d := cpodv1.DataSetStorage{
		TypeMeta: metav1.TypeMeta{},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: v1beta1.CPOD_NAMESPACE,
			Name:      datasetStorageName,
		},
		Spec: cpodv1.DataSetStorageSpec{
			DatasetType: "oss",
			DatasetName: datasetName,
			PVC:         pvcName,
		},
	}
	err := s.kubeClient.Create(ctx, &d)
	return err
}

func (s *SyncJob) createPVC(ctx context.Context, pvcName, pvcSize, storageClass string) error {
	volumeMode := v1.PersistentVolumeFilesystem
	pvc := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: v1beta1.CPOD_NAMESPACE},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteMany,
			},
			Resources: v1.ResourceRequirements{
				Requests: map[v1.ResourceName]resource.Quantity{
					v1.ResourceStorage: resource.MustParse(pvcSize),
				},
			},
			StorageClassName: &storageClass,
			VolumeMode:       &volumeMode,
		},
	}
	err := s.kubeClient.Create(ctx, &pvc)
	return err

}

func (s *SyncJob) createDownloaderJob(ctx context.Context, tp, pvcName, downloadJobName, storageID, storageSize, ossPath, ossAK, ossSK string) error {
	pint32 := func(i int32) *int32 {
		return &i
	}
	pint64 := func(i int64) *int64 {
		return &i
	}
	pstring := func(i batchv1.CompletionMode) *batchv1.CompletionMode {
		return &i
	}
	j := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: v1beta1.CPOD_NAMESPACE,
			Name:      downloadJobName,
			Labels:    map[string]string{v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource},
		},
		Spec: batchv1.JobSpec{
			Parallelism:  pint32(1),
			Completions:  pint32(1),
			BackoffLimit: pint32(6),
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "data-volume",
							VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvcName,
								ReadOnly:  false,
							}},
						},
					},
					Containers: []v1.Container{
						{
							Name:  tp + "-downloader",
							Image: "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:v0.0.7",
							Args: []string{
								"-g",
								"cpod.cpod",
								"-v",
								"v1",
								"-p",
								tp + "storages",
								"-n",
								"cpod",
								"--name",
								storageID,
								"oss",
								ossPath,
								"-t",
								storageSize,
								"--endpoint",
								"https://oss-cn-beijing.aliyuncs.com",
								"--access_id",
								ossAK,
								"--access_key",
								ossSK,
							},
							VolumeMounts: []v1.VolumeMount{
								{
									MountPath: "/data",
									Name:      "data-volume",
								},
							},
							ImagePullPolicy: v1.PullIfNotPresent,
						},
					},
					RestartPolicy:                 "Never",
					TerminationGracePeriodSeconds: pint64(30),
					DNSPolicy:                     "ClusterFirst",
					ServiceAccountName:            "sa-downloader",
					ImagePullSecrets: []v1.LocalObjectReference{
						{Name: "aliyun-enterprise-registry"},
					},
				},
			},
			CompletionMode: pstring(batchv1.NonIndexedCompletion),
		},
	}
	err := s.kubeClient.Create(ctx, &j)
	return err
}
