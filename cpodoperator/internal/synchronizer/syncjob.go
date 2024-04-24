package synchronizer

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/pointer"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
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
	mjj map[string]sxwl.PortalJupyterLabJob
	mu  *sync.RWMutex
}

type SyncJob struct {
	kubeClient           client.Client
	scheduler            sxwl.Scheduler
	createFailedJobs     jobBuffer
	preparingJobs        jobBuffer
	logger               logr.Logger
	uploadTrainedModel   bool
	autoDownloadResource bool
	inferImage           string
}

func NewSyncJob(kubeClient client.Client, scheduler sxwl.Scheduler, logger logr.Logger, uploadTrainedModel, autoDownloadReource bool, inferImage string) *SyncJob {
	return &SyncJob{
		kubeClient: kubeClient,
		scheduler:  scheduler,
		createFailedJobs: jobBuffer{
			mtj: map[string]sxwl.PortalTrainningJob{},
			mij: map[string]sxwl.PortalInferenceJob{},
			mjj: map[string]sxwl.PortalJupyterLabJob{},
			mu:  new(sync.RWMutex),
		},
		preparingJobs: jobBuffer{
			mtj: map[string]sxwl.PortalTrainningJob{},
			mij: map[string]sxwl.PortalInferenceJob{},
			mjj: map[string]sxwl.PortalJupyterLabJob{},
			mu:  new(sync.RWMutex),
		},
		logger:               logger,
		uploadTrainedModel:   uploadTrainedModel,
		autoDownloadResource: autoDownloadReource,
		inferImage:           inferImage,
	}
}

func jobTypeCheck(jobtype string) (v1beta1.JobType, bool) {
	switch strings.ToLower(jobtype) {
	case string(v1beta1.JobTypeMPI):
		return v1beta1.JobTypeMPI, true
	case string(v1beta1.JobTypePytorch):
		return v1beta1.JobTypePytorch, true
	}
	return "", false
}

// first retrieve twn job sets , portal job set and cpod job set
// for jobs in portal not in cpod , create it
// for jobs in cpod not in portal , if it's running , delete it
func (s *SyncJob) Start(ctx context.Context) {
	s.logger.Info("sync job")

	portalTrainningJobs, portalInferenceJobs, portalJupyterLabJobs, users, err := s.scheduler.GetAssignedJobList()
	if err != nil {
		s.logger.Error(err, "failed to list job")
		return
	}
	s.logger.Info("assigned trainning job", "jobs", portalTrainningJobs)
	s.logger.Info("assigned inference job", "jobs", portalInferenceJobs)
	s.syncUsers(ctx, users)
	s.processTrainningJobs(ctx, users, portalTrainningJobs)
	s.processFinetune(ctx, users, portalTrainningJobs)
	s.processInferenceJobs(ctx, users, portalInferenceJobs)
	s.processJupyterLabJobs(ctx, portalJupyterLabJobs)

}

func (s *SyncJob) syncUsers(ctx context.Context, userIDs []sxwl.UserID) {
	var users v1.NamespaceList
	err := s.kubeClient.List(ctx, &users, client.HasLabels{
		v1beta1.CPodUserNamespaceLabel,
	})
	if err != nil {
		s.logger.Error(err, "failed to list users")
		return
	}
	for _, user := range userIDs {
		exists := false
		for _, u := range users.Items {
			if u.Name == string(user) {
				exists = true
			}
		}
		if !exists {
			newUserNamespace := v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: string(user),
					Labels: map[string]string{
						v1beta1.CPodUserNamespaceLabel: string(user),
					},
				},
			}
			if err = s.kubeClient.Create(ctx, &newUserNamespace); err != nil {
				s.logger.Error(err, "failed to create user namespace", "user", user)
			} else {
				s.logger.Info("user created", "user", newUserNamespace)
			}
		}
	}
}

func (s *SyncJob) processFinetune(ctx context.Context, userIDs []sxwl.UserID, portaljobs []sxwl.PortalTrainningJob) {
	for _, user := range userIDs {
		var finetunes v1beta1.FineTuneList
		err := s.kubeClient.List(ctx, &finetunes, &client.MatchingLabels{
			v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
		}, client.InNamespace(user))
		if err != nil {
			s.logger.Error(err, "failed to list finetunejob")
		}
		for _, job := range portaljobs {
			if string(job.UserID) != string(user) {
				continue
			}
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
				// create
				newJob := v1beta1.FineTune{
					ObjectMeta: metav1.ObjectMeta{
						Name:      job.JobName,
						Namespace: string(user),
						Labels: map[string]string{
							v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
							v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
						},
						Annotations: map[string]string{
							v1beta1.CPodModelstorageNameAnno: job.TrainedModelName,
						},
					},
					Spec: v1beta1.FineTuneSpec{
						Model:          job.PretrainModelName,
						DatasetStorage: job.DatasetId,
						Upload:         s.uploadTrainedModel,
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
}

func (s *SyncJob) processTrainningJobs(ctx context.Context, userIDs []sxwl.UserID, portaljobs []sxwl.PortalTrainningJob) {
	for _, user := range userIDs {
		s.logger.Info("sync trainning job of user", "user", user)
		var cpodTrainningJobs v1beta1.CPodJobList
		err := s.kubeClient.List(ctx, &cpodTrainningJobs, &client.MatchingLabels{
			v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
		}, client.InNamespace(user))
		if err != nil {
			s.logger.Error(err, "failed to list trainningjob")
			return
		}

		for _, job := range portaljobs {
			if string(job.UserID) != string(user) {
				continue
			}

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

				if job.PretrainModelId != "" && s.autoDownloadResource {
					// 判断指定的预训练模型是否存在
					exists, done, err := s.checkModelExistence(ctx, string(user), job.PretrainModelId, job.PretrainModelIsPublic)
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
						// create PVC
						ossAK := os.Getenv("AK")
						ossSK := os.Getenv("AS")
						storageClassName := os.Getenv("STORAGECLASS")
						ossPath := ResourceToOSSPath(Model, job.PretrainModelName)
						pvcName := ModelPVCName(ossPath)
						// storageName := ModelCRDName(ossPath)
						modelSize := fmt.Sprintf("%d", job.PretrainModelSize)
						// pvcsize is 1.2 * modelsize
						pvcSize := fmt.Sprintf("%dMi", job.PretrainModelSize*12/10/1024/1024)
						err := s.createPVC(ctx, pvcName, pvcSize, storageClassName)
						if err != nil {
							s.logger.Error(err, "create pvc failed", "jobname", job.JobName, "modelid", job.PretrainModelId)
							continue
						} else {
							s.logger.Info("pvc created", "jobname", job.JobName, "modelid", job.PretrainModelId)
						}
						// create ModelStorage
						err = s.createModelStorage(ctx, job.PretrainModelId, job.PretrainModelName, pvcName)
						if err != nil {
							s.logger.Error(err, "create modelstorage failed", "jobname", job.JobName, "modelid", job.PretrainModelId)
							continue
						} else {
							s.logger.Info("modelstorage created", "jobname", job.JobName, "modelid", job.PretrainModelId)
						}
						// create DownloaderJob
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
				if job.DatasetId != "" && s.autoDownloadResource {
					// 判断指定的预训练模型是否存在
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
						// create PVC
						ossAK := os.Getenv("AK")
						ossSK := os.Getenv("AS")
						storageClassName := os.Getenv("STORAGECLASS")
						ossPath := ResourceToOSSPath(Dataset, job.DatasetName)
						pvcName := DatasetPVCName(ossPath)
						datasetSize := fmt.Sprintf("%d", job.DatasetSize)
						// pvcsize is 1.2 * datasetsize
						pvcSize := fmt.Sprintf("%dMi", job.DatasetSize*12/10/1024/1024)
						err := s.createPVC(ctx, pvcName, pvcSize, storageClassName)
						if err != nil {
							s.logger.Error(err, "create pvc failed", "jobname", job.JobName, "datasetid", job.DatasetId)
							continue
						} else {
							s.logger.Info("pvc created", "jobname", job.JobName, "datasetid", job.DatasetId)
						}
						// create DatasetStorage
						err = s.createDatasetStorage(ctx, job.DatasetId, job.DatasetName, pvcName)
						if err != nil {
							s.logger.Error(err, "create datasetstorage failed", "jobname", job.JobName, "datasetid", job.DatasetId)
							continue
						} else {
							s.logger.Info("datasetstorage created", "jobname", job.JobName, "datasetid", job.DatasetId)
						}
						// create DownloaderJob
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
						Namespace: string(user),
						Name:      job.JobName,
						Labels: map[string]string{
							v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
							v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
						},
						Annotations: map[string]string{
							v1beta1.CPodModelstorageNameAnno: job.TrainedModelName,
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
}

func (s *SyncJob) processInferenceJobs(ctx context.Context, userIDs []sxwl.UserID, portaljobs []sxwl.PortalInferenceJob) {
	for _, user := range userIDs {
		var cpodInferenceJobs v1beta1.InferenceList
		err := s.kubeClient.List(ctx, &cpodInferenceJobs, &client.MatchingLabels{
			v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
		}, client.InNamespace(user))
		if err != nil {
			s.logger.Error(err, "failed to list inference job")
			return
		}

		for _, job := range portaljobs {
			if string(job.UserID) != string(user) {
				continue
			}
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
				// create
				newJob := v1beta1.Inference{
					ObjectMeta: metav1.ObjectMeta{
						// TODO: create namespace for different tenant
						Namespace: string(user),
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
										Image: s.inferImage,
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
												"nvidia.com/gpu":  resource.MustParse(strconv.FormatInt(job.GpuNumber, 10)),
											},
											Requests: map[v1.ResourceName]resource.Quantity{
												v1.ResourceCPU:    resource.MustParse("4"),
												v1.ResourceMemory: resource.MustParse("50Gi"),
												"nvidia.com/gpu":  resource.MustParse(strconv.FormatInt(job.GpuNumber, 10)),
											},
										},
									},
								},
								NodeSelector: map[string]string{
									"nvidia.com/gpu.product": job.GpuType,
								},
							},
						},
					},
				}
				if job.GpuNumber > 1 {
					cudaDevices := ""
					for i := 0; i < int(job.GpuNumber); i++ {
						if i == int(job.GpuNumber)-1 {
							cudaDevices = cudaDevices + strconv.Itoa(i) + ","
						} else {
							cudaDevices = cudaDevices + strconv.Itoa(i) + ","
						}
					}
					newJob.Spec.Predictor.PodSpec.Containers[0].Env = append(newJob.Spec.Predictor.PodSpec.Containers[0].Env, v1.EnvVar{
						Name:  "CUDA_VISIBLE_DEVICES",
						Value: cudaDevices,
					})

					newJob.Spec.Predictor.PodSpec.Containers[0].Command = append(newJob.Spec.Predictor.PodSpec.Containers[0].Command, "--infer_backend", "vllm", "--vllm_enforce_eager")
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
func (s *SyncJob) checkModelExistence(ctx context.Context, namespace, m string, modelType bool) (bool, bool, error) {
	var ms cpodv1.ModelStorage
	if modelType {
		var publicMs cpodv1.ModelStorage
		// 判断用户命名空间公共模型是否已经拷贝
		err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: m + "-public"}, &ms)
		if err != nil {
			if kerrors.IsNotFound(err) {
				// 还未拷贝
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: m}, &publicMs); err != nil {
					if kerrors.IsNotFound(err) {
						// 公共模型不存在
						return false, false, nil
					}
					return false, false, err
				}
				// 创建用户命名空间的公共模型的拷贝
				var publicMsPVC v1.PersistentVolumeClaim
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: publicMs.Spec.PVC}, &publicMsPVC); err != nil {
					return false, false, err
				}
				var publicMsPV v1.PersistentVolume
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: publicMsPVC.Spec.VolumeName}, &publicMsPV); err != nil {
					return false, false, err
				}
				// 创建pv
				pvCopy := publicMsPV.DeepCopy()
				pvCopy.Name = pvCopy.Name + "-" + namespace
				if err := s.kubeClient.Create(ctx, pvCopy); err != nil {
					return false, false, err
				}
				// 创建pvc
				pvcCopy := publicMsPVC.DeepCopy()
				pvcCopy.Name = pvcCopy.Name
				pvcCopy.Namespace = namespace
				pvcCopy.Spec.VolumeName = pvCopy.Name
				if err := s.kubeClient.Create(ctx, pvcCopy); err != nil {
					return false, false, err
				}
				// 创建modelstorage
				modelStorageCopy := publicMs.DeepCopy()
				modelStorageCopy.Name = modelStorageCopy.Name + "-public"
				modelStorageCopy.Namespace = namespace
				modelStorageCopy.Spec.PVC = pvcCopy.Name
				modelStorageCopy.Labels[v1beta1.CPODStorageCopyLable] = "true"
				if err := s.kubeClient.Create(ctx, modelStorageCopy); err != nil {
					return false, false, err
				}
				// TODO: update modelstorage status
				modelStorageCopy.Status.Phase = "done"
				if err := s.kubeClient.Status().Update(ctx, modelStorageCopy); err != nil {
					return false, false, err
				}
				return true, true, nil
			}
			return false, false, err
		}
		return true, true, nil
	}
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
	if namespace == "public" {
		var publicDs cpodv1.DataSetStorage
		// 判断用户命名空间公共数据集是否已经拷贝
		err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: d + "-public"}, &ds)
		if err != nil {
			if kerrors.IsNotFound(err) {
				// 还未拷贝
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: d}, &publicDs); err != nil {
					if kerrors.IsNotFound(err) {
						// 公共数据集不存在
						return false, false, nil
					}
					return false, false, err
				}
				// 创建用户命名空间的公共数据集的拷贝
				var publicDsPVC v1.PersistentVolumeClaim
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: publicDs.Spec.PVC}, &publicDsPVC); err != nil {
					return false, false, err
				}
				var publicDsPV v1.PersistentVolume
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: "public", Name: publicDsPVC.Spec.VolumeName}, &publicDsPV); err != nil {
					return false, false, err
				}
				// 创建pv
				pvCopy := publicDsPV.DeepCopy()
				pvCopy.Name = pvCopy.Name + "-" + namespace
				if err := s.kubeClient.Create(ctx, pvCopy); err != nil {
					return false, false, err
				}
				// 创建pvc
				pvcCopy := publicDsPVC.DeepCopy()
				pvcCopy.Name = pvcCopy.Name
				pvcCopy.Namespace = namespace
				pvcCopy.Spec.VolumeName = pvCopy.Name
				if err := s.kubeClient.Create(ctx, pvcCopy); err != nil {
					return false, false, err
				}
				// 创建modelstorage
				datasetStorageCopy := publicDs.DeepCopy()
				datasetStorageCopy.Name = datasetStorageCopy.Name + "-public"
				datasetStorageCopy.Namespace = namespace
				datasetStorageCopy.Spec.PVC = pvc
				datasetStorageCopy.Labels[v1beta1.CPODStorageCopyLable] = "true"
				if err := s.kubeClient.Create(ctx, datasetStorageCopy); err != nil {
					return false, false, err
				}
				// TODO: update modelstorage status
				datasetStorageCopy.Status.Phase = "done"
				if err := s.kubeClient.Status().Update(ctx, datasetStorageCopy); err != nil {
					return false, false, err
				}
				return true, true, nil
			}
			return false, false, err
		}
		return true, true, nil
	}
	err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: d}, &ds)
	if err != nil {
		if kerrors.IsNotFound(err) {
			return false, false, nil
		}
		return false, false, err
	}
	// TODO @sxwl-donggang: 公共模型的拷贝
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

func (s *SyncJob) processJupyterLabJobs(ctx context.Context, portalJobs []sxwl.PortalJupyterLabJob) error {
	// 1. 获取当前Kubernetes集群中的JupyterLab任务列表
	var currentJupyterLabJobs appsv1.StatefulSetList
	err := s.kubeClient.List(ctx, &currentJupyterLabJobs, &client.MatchingLabels{
		"app": "jupyterlab",
	})
	if err != nil {
		s.logger.Error(err, "failed to list JupyterLab jobs")
		return err
	}

	// 2. 遍历并同步JupyterLab任务
	for _, job := range portalJobs {
		found := false
		for _, currentJob := range currentJupyterLabJobs.Items {
			if currentJob.Name == job.JobName {
				found = true
			}
		}

		if !found {
			// 创建新的JupyterLab任务
			ss, err := s.createJupyterLabStatefulSet(ctx, job)
			if err != nil {
				s.addCreateFailedJupyterLabJob(job)
				s.logger.Error(err, "failed to create JupyterLab StatefulSet")
				return err
			} else {
				s.deleteCreateFailedJupyterLabJob(job.JobName)
				s.logger.Info("JupyterLab StatefulSet created", "statefulset", ss.Name)
			}

			ownerRef := metav1.OwnerReference{
				APIVersion: "apps/v1",
				Kind:       "StatefulSet",
				Name:       ss.Name,
				UID:        ss.UID,
			}

			svc, err := s.createJupyterLabService(ctx, job, ownerRef)
			if err != nil {
				s.logger.Error(err, "failed to create JupyterLab service")
				return err
			}
			s.logger.Info("JupyterLab service created", "service", svc.Name)

			ing, err := s.createJupyterLabIngress(ctx, job, svc, ownerRef)
			if err != nil {
				s.logger.Error(err, "failed to create JupyterLab ingress")
				return err
			}
			s.logger.Info("JupyterLab ingress created", "ingress", ing.Name)
		}
	}

	// 3. 删除不再需要的JupyterLab任务
	for _, currentJob := range currentJupyterLabJobs.Items {
		exists := false
		for _, job := range portalJobs {
			if currentJob.Name == job.JobName {
				exists = true
				break
			}
		}

		if !exists {
			// 删除当前的JupyterLab任务
			if err := s.kubeClient.Delete(ctx, &currentJob); err != nil {
				s.logger.Error(err, "failed to delete JupyterLab job", "job", currentJob)
			}
		}
	}

	return nil
}

func (s *SyncJob) createJupyterLabStatefulSet(ctx context.Context, job sxwl.PortalJupyterLabJob) (*appsv1.StatefulSet, error) {
	volumeMounts := []v1.VolumeMount{
		{
			Name:      "workspace",
			MountPath: "/workspace",
		},
	}
	volumes := []v1.Volume{
		{
			Name: "workspace",
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: job.JobName + "-workspace",
				},
			},
		},
	}

	// 处理预训练模型的挂载点
	for _, pm := range *job.PretrainedModels {
		var modelStorage cpodv1.ModelStorage
		if err := s.kubeClient.Get(ctx, client.ObjectKey{Name: pm.PretrainedModelId, Namespace: v1beta1.CPOD_NAMESPACE}, &modelStorage); err != nil {
			return nil, fmt.Errorf("failed to get ModelStorage %s: %w", pm.PretrainedModelId, err)
		}

		// 添加挂载点
		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      pm.PretrainedModelId,
			MountPath: pm.PretrainedModelPath,
		})

		// 添加对应的卷
		volumes = append(volumes, v1.Volume{
			Name: pm.PretrainedModelId,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: modelStorage.Spec.PVC,
					ReadOnly:  true,
				},
			},
		})
	}

	// 2. 创建StatefulSet
	ss := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      job.JobName,
			Namespace: v1beta1.CPOD_NAMESPACE,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "jupyterlab"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "jupyterlab"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  job.JobName,
							Image: "dockerhub.kubekey.local/kubesphereio/jupyterlab:v5",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse(job.CPUCount),
									v1.ResourceMemory: resource.MustParse(job.Memory),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse(job.CPUCount),
									v1.ResourceMemory: resource.MustParse(job.Memory),
									"nvidia.com/gpu":  *resource.NewQuantity(int64(job.GPUCount), resource.DecimalSI),
								},
							},
							Env: []v1.EnvVar{
								{
									Name:  "JUPYTER_TOKEN",
									Value: job.JobName,
								},
							},
							Command: []string{
								"jupyter",
								"lab",
								fmt.Sprintf("--ServerApp.base_url=/jupyterlab/%s/", job.JobName),
								"--allow-root",
								"--ip=0.0.0.0",
							},
							VolumeMounts: volumeMounts,
						},
					},
					Volumes: volumes,
				},
			},
			VolumeClaimTemplates: []v1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: job.JobName + "-workspace",
					},
					Spec: v1.PersistentVolumeClaimSpec{
						AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceStorage: resource.MustParse(job.DataVolumeSize),
							},
						},
					},
				},
			},
		},
	}

	if job.GPUProduct != "" {
		ss.Spec.Template.Spec.NodeSelector = map[string]string{
			"nvidia.com/gpu.product": job.GPUProduct,
		}
	}

	// 3. 创建StatefulSet
	if err := s.kubeClient.Create(ctx, ss); err != nil {
		return nil, fmt.Errorf("failed to create StatefulSet %s: %w", job.JobName, err)
	}

	return ss, nil
}

func (s *SyncJob) createJupyterLabService(ctx context.Context, job sxwl.PortalJupyterLabJob, ownerRef metav1.OwnerReference) (*v1.Service, error) {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            job.JobName + "-svc",
			Namespace:       v1beta1.CPOD_NAMESPACE,
			OwnerReferences: []metav1.OwnerReference{ownerRef},
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{
				"app": job.JobName,
			},
			Ports: []v1.ServicePort{
				{
					Port:       8888,
					TargetPort: intstr.FromInt(8888),
				},
			},
			Type: v1.ServiceTypeClusterIP, // 使用ClusterIP以便在集群内部访问
		},
	}

	if err := s.kubeClient.Create(ctx, svc); err != nil {
		return nil, fmt.Errorf("failed to create service for JupyterLab %s: %v", job.JobName, err)
	}

	return svc, nil
}

func (s *SyncJob) createJupyterLabIngress(ctx context.Context, job sxwl.PortalJupyterLabJob, svc *v1.Service, ownerRef metav1.OwnerReference) (*networkingv1.Ingress, error) {
	pathType := networkingv1.PathTypeImplementationSpecific
	ing := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:            job.JobName + "-ing",
			Namespace:       v1beta1.CPOD_NAMESPACE,
			OwnerReferences: []metav1.OwnerReference{ownerRef},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/jupyterlab/" + job.JobName + "/$2",
			},
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: pointer.StringPtr("nginx"),
			Rules: []networkingv1.IngressRule{
				{
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									PathType: &pathType,
									Path:     "/jupyterlab/" + job.JobName + "(/|$)(.*)",
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: job.JobName + "-svc",
											Port: networkingv1.ServiceBackendPort{
												Number: 8888,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	if err := s.kubeClient.Create(ctx, ing); err != nil {
		return nil, fmt.Errorf("failed to create ingress for JupyterLab %s: %v", job.InstanceName, err)
	}

	return ing, nil
}

func (s *SyncJob) getCreateFailedJupyterLabJobs() []sxwl.PortalJupyterLabJob {
	res := []sxwl.PortalJupyterLabJob{}
	s.createFailedJobs.mu.RLock()
	defer s.createFailedJobs.mu.RUnlock()
	for _, v := range s.createFailedJobs.mjj {
		res = append(res, v)
	}
	return res
}

func (s *SyncJob) addCreateFailedJupyterLabJob(j sxwl.PortalJupyterLabJob) {
	if _, ok := s.createFailedJobs.mjj[j.JobName]; ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	s.createFailedJobs.mjj[j.JobName] = j
}

// 如果任务创建成功了，将其从失败任务列表中删除
func (s *SyncJob) deleteCreateFailedJupyterLabJob(j string) {
	if _, ok := s.createFailedJobs.mjj[j]; !ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	delete(s.createFailedJobs.mjj, j)
}
