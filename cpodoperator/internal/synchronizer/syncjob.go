package synchronizer

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
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
	s.logger.Info("assigned trainning job", "jobs", portalTrainningJobs, "users", users)
	s.logger.Info("assigned inference job", "jobs", portalInferenceJobs)
	s.syncUsers(ctx, users)
	s.processTrainningJobs(ctx, users, portalTrainningJobs)
	s.processFinetune(ctx, users, portalTrainningJobs)
	s.processInferenceJobs(ctx, users, portalInferenceJobs)
	s.processJupyterLabJobs(ctx, portalJupyterLabJobs, users)

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
	for _, userNs := range users.Items {
		var clusterrolebinding rbacv1.ClusterRoleBinding
		if err := s.kubeClient.Get(ctx, types.NamespacedName{Name: userNs.Name}, &clusterrolebinding); err != nil {
			if kerrors.IsNotFound(err) {
				newClusterRoleBinding := rbacv1.ClusterRoleBinding{
					ObjectMeta: metav1.ObjectMeta{
						Name: userNs.Name,
					},
					Subjects: []rbacv1.Subject{
						{
							Kind:      "ServiceAccount",
							Name:      "default",
							Namespace: userNs.Namespace,
						},
					},
					RoleRef: rbacv1.RoleRef{
						Kind:     "ClusterRole",
						Name:     "cluster-admin",
						APIGroup: "rbac.authorization.k8s.io",
					},
				}
				if err = s.kubeClient.Create(ctx, &newClusterRoleBinding); err != nil {
					s.logger.Error(err, "failed to create clusterrolebinding", "user", userNs.Name)
				} else {
					s.logger.Info("clusterrolebinding created", "user", userNs.Name)
				}
			} else {
				s.logger.Error(err, "failed to get clusterrolebinding", "user", userNs.Name)
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
			if job.UserID != string(user) {
				continue
			}
			if job.JobType != "Finetune" {
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
				newJob := v1beta1.FineTune{
					ObjectMeta: metav1.ObjectMeta{
						Name:      job.JobName,
						Namespace: string(user),
						Labels: map[string]string{
							v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
							v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
						},
						Annotations: map[string]string{
							v1beta1.CPodModelstorageNameAnno:     job.TrainedModelName,
							v1beta1.CPodDatasetlReadableNameAnno: job.DatasetName,
							v1beta1.CPodDatasetSizeAnno:          fmt.Sprintf("%d", job.DatasetSize),
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

	}
	var totalFinetunes v1beta1.FineTuneList
	if err := s.kubeClient.List(ctx, &totalFinetunes, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	}); err != nil {
		s.logger.Error(err, "failed to list all finetunes")
		return
	}

	for _, finetune := range totalFinetunes.Items {
		exists := false
		for _, job := range portaljobs {
			if finetune.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			if err := s.kubeClient.Delete(ctx, &finetune); err != nil {
				s.logger.Error(err, "failed to delete finetune")
				return
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
			if job.UserID != string(user) {
				continue
			}

			if job.JobType == "Finetune" {
				continue
			}

			exists := false
			for _, cpodTrainningJob := range cpodTrainningJobs.Items {
				if cpodTrainningJob.Name == job.JobName {
					exists = true
				}
			}
			if !exists {
				s.logger.Info("trainning job", "trainning", job.JobName, "job", job)
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
							v1beta1.CPodModelstorageNameAnno:          job.TrainedModelName,
							v1beta1.CPodPreTrainModelReadableNameAnno: job.PretrainModelName,
							v1beta1.CPodPreTrainModelSizeAnno:         fmt.Sprintf("%d", job.PretrainModelSize),
							v1beta1.CPodPreTrainModelTemplateAnno:     job.PretrainModelTemplate,
							v1beta1.CPodDatasetlReadableNameAnno:      job.DatasetName,
							v1beta1.CPodDatasetSizeAnno:               fmt.Sprintf("%d", job.DatasetSize),
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
					s.logger.Error(err, "failed to create trainningjob", "job", newJob)
				} else {
					s.logger.Info("trainningjob created", "job", newJob)
				}
			}
		}
	}

	var totalCPodJobs v1beta1.CPodJobList
	if err := s.kubeClient.List(ctx, &totalCPodJobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	}); err != nil {
		s.logger.Error(err, "failed to list all cpod jobs")
		return
	}

	for _, cpodTrainningJob := range totalCPodJobs.Items {
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
			if err := s.kubeClient.Delete(ctx, &cpodTrainningJob); err != nil {
				s.logger.Error(err, "failed to delete trainningjob")
				return
			}
			s.logger.Info("trainningjob deleted", "jobid", cpodTrainningJob.Name)
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
		s.logger.Info("DEBUG sync inference job of user", "user", user, "jobs", portaljobs, "existing", cpodInferenceJobs.Items)

		for _, job := range portaljobs {
			if job.UserID != string(user) {
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
						Annotations: map[string]string{
							v1beta1.CPodPreTrainModelReadableNameAnno: job.ModelName,
							v1beta1.CPodPreTrainModelSizeAnno:         fmt.Sprintf("%d", job.ModelSize),
							v1beta1.CPodPreTrainModelTemplateAnno:     job.Template,
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
											"--infer_backend",
											"vllm",
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
						ModelIsPublic: job.ModelIsPublic,
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
					s.logger.Error(err, "failed to create inference job", "job", newJob)
				} else {
					s.logger.Info("inference job created", "job", newJob)
				}
			}
		}

	}

	var totalInferencs v1beta1.InferenceList
	if err := s.kubeClient.List(ctx, &totalInferencs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	}); err != nil {
		s.logger.Error(err, "failed to list all inference jobs")
		return
	}

	for _, cpodInferenceJob := range totalInferencs.Items {
		exists := false
		for _, job := range portaljobs {
			if cpodInferenceJob.Name == job.ServiceName {
				exists = true
			}
		}
		if !exists {
			if err := s.kubeClient.Delete(ctx, &cpodInferenceJob); err != nil {
				s.logger.Error(err, "failed to delete inference job")
				return
			}
			s.logger.Info("inference job deleted", "jobid", cpodInferenceJob.Name)
		}
	}
}

// 检查模型是否存在
func (s *SyncJob) checkModelExistence(ctx context.Context, namespace, m string, modelType bool) (bool, bool, error) {
	var ms cpodv1.ModelStorage
	if modelType {
		var publicMs cpodv1.ModelStorage
		// 判断用户命名空间公共模型是否已经拷贝
		err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: m + v1beta1.CPodPublicStorageSuffix}, &ms)
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
				pvCopy.ResourceVersion = ""
				pvCopy.UID = ""
				pvName := pvCopy.Name + "-" + namespace
				if len(pvName) > 63 {
					pvName = pvName[:63]
				}
				if strings.HasSuffix(pvName, "-") {
					pvName = pvName[:len(pvName)-1]
				}
				pvCopy.Name = pvName
				pvCopy.Spec.CSI.VolumeHandle = pvName
				if err := s.kubeClient.Create(ctx, pvCopy); err != nil && !kerrors.IsAlreadyExists(err) {
					return false, false, err
				}
				// 创建pvc
				pvcCopy := publicMsPVC.DeepCopy()
				pvcCopy.Name = pvcCopy.Name + v1beta1.CPodPublicStorageSuffix
				pvcCopy.Namespace = namespace
				pvcCopy.Spec.VolumeName = pvCopy.Name
				pvcCopy.ResourceVersion = ""
				pvcCopy.UID = ""
				if err := s.kubeClient.Create(ctx, pvcCopy); err != nil && !kerrors.IsAlreadyExists(err) {
					return false, false, err
				}
				// 创建modelstorage
				modelStorageCopy := publicMs.DeepCopy()
				modelStorageCopy.Name = modelStorageCopy.Name + v1beta1.CPodPublicStorageSuffix
				modelStorageCopy.Namespace = namespace
				modelStorageCopy.Spec.PVC = pvcCopy.Name
				modelStorageCopy.ResourceVersion = ""
				modelStorageCopy.UID = ""
				if modelStorageCopy.Labels == nil {
					modelStorageCopy.Labels = make(map[string]string)
				}
				modelStorageCopy.Labels[v1beta1.CPODStorageCopyLable] = "true"
				if err := s.kubeClient.Create(ctx, modelStorageCopy); err != nil && !kerrors.IsAlreadyExists(err) {
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
func (s *SyncJob) checkDatasetExistence(ctx context.Context, namespace, d string, datasetType bool) (bool, bool, error) {
	var ds cpodv1.DataSetStorage
	if datasetType {
		var publicDs cpodv1.DataSetStorage
		// 判断用户命名空间公共数据集是否已经拷贝
		err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: d + v1beta1.CPodPublicStorageSuffix}, &ds)
		if err != nil {
			if kerrors.IsNotFound(err) {
				// 还未拷贝
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: d}, &publicDs); err != nil {
					if kerrors.IsNotFound(err) {
						// 公共数据集不存在
						return false, false, nil
					}
					return false, false, err
				}
				// 创建用户命名空间的公共数据集的拷贝
				var publicDsPVC v1.PersistentVolumeClaim
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: publicDs.Spec.PVC}, &publicDsPVC); err != nil {
					return false, false, err
				}
				var publicDsPV v1.PersistentVolume
				if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: v1beta1.CPodPublicNamespace, Name: publicDsPVC.Spec.VolumeName}, &publicDsPV); err != nil {
					return false, false, err
				}
				// 创建pv
				pvCopy := publicDsPV.DeepCopy()
				pvName := pvCopy.Name + "-" + namespace
				if len(pvName) > 63 {
					pvName = pvName[:63]
				}
				if strings.HasSuffix(pvName, "-") {
					pvName = pvName[:len(pvName)-1]
				}
				pvCopy.Name = pvName
				pvCopy.Spec.CSI.VolumeHandle = pvName
				pvCopy.ResourceVersion = ""
				pvCopy.UID = ""
				if err := s.kubeClient.Create(ctx, pvCopy); err != nil && !kerrors.IsAlreadyExists(err) {
					return false, false, err
				}
				// 创建pvc
				pvcCopy := publicDsPVC.DeepCopy()
				pvcCopy.Name = pvcCopy.Name + v1beta1.CPodPublicStorageSuffix
				pvcCopy.Namespace = namespace
				pvcCopy.Spec.VolumeName = pvCopy.Name
				pvcCopy.ResourceVersion = ""
				pvCopy.UID = ""
				if err := s.kubeClient.Create(ctx, pvcCopy); err != nil && !kerrors.IsAlreadyExists(err) {
					return false, false, err
				}
				// 创建modelstorage
				datasetStorageCopy := publicDs.DeepCopy()
				datasetStorageCopy.Name = datasetStorageCopy.Name + v1beta1.CPodPublicStorageSuffix
				datasetStorageCopy.Namespace = namespace
				datasetStorageCopy.Spec.PVC = pvcCopy.Name
				datasetStorageCopy.ResourceVersion = ""
				datasetStorageCopy.UID = ""
				if datasetStorageCopy.Labels == nil {
					datasetStorageCopy.Labels = make(map[string]string)
				}
				datasetStorageCopy.Labels[v1beta1.CPODStorageCopyLable] = "true"
				if err := s.kubeClient.Create(ctx, datasetStorageCopy); err != nil && !kerrors.IsAlreadyExists(err) {
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

func (s *SyncJob) processJupyterLabJobs(ctx context.Context, portalJobs []sxwl.PortalJupyterLabJob, userIDs []sxwl.UserID) {
	// 1. 获取当前Kubernetes集群中的JupyterLab任务列表
	for _, UserID := range userIDs {
		var currentJupyterLabJobs v1beta1.JuypterLabList
		err := s.kubeClient.List(ctx, &currentJupyterLabJobs, &client.MatchingLabels{
			v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
		}, client.InNamespace(UserID))
		if err != nil {
			s.logger.Error(err, "failed to list JupyterLabs")
			return
		}
		s.logger.Info("DEBUG sync juypterlab of user", "user", userIDs, "juypterlabs", portalJobs, "existing", currentJupyterLabJobs.Items)

		// 2. 遍历并同步JupyterLab任务
		for _, job := range portalJobs {
			if job.UserID != string(UserID) {
				continue
			}
			found := false
			for _, currentJob := range currentJupyterLabJobs.Items {
				if currentJob.Name == job.JobName {
					found = true
				}
			}

			if !found {
				// 创建新的JupyterLab任务
				err := s.createJupyterLab(ctx, string(UserID), job)
				if err != nil {
					s.logger.Error(err, "failed to create JupyterLab StatefulSet")
				}
			}
		}

	}
	var totalJuypterlabJobs v1beta1.JuypterLabList
	if err := s.kubeClient.List(ctx, &totalJuypterlabJobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	}); err != nil {
		s.logger.Error(err, "failed to list JupyterLabs")
		return
	}
	for _, currentJob := range totalJuypterlabJobs.Items {
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
}

func (s *SyncJob) createJupyterLab(ctx context.Context, namespace string, job sxwl.PortalJupyterLabJob) error {
	// 处理预训练模型的挂载点
	models := []v1beta1.Model{}
	for _, pm := range *job.PretrainedModels {
		_, done, err := s.checkModelExistence(ctx, namespace, pm.PretrainedModelId, pm.PretrainedModelIsPublic)
		if err != nil {
			s.logger.Error(err, "failed to check model existence", "modelid", pm.PretrainedModelId)
			continue
		}
		if !done {
			s.logger.Info("Model is preparing.", "jobname", job.JobName, "modelid", pm.PretrainedModelId)
			continue
		}
		modelID := pm.PretrainedModelId
		if pm.PretrainedModelIsPublic {
			modelID = modelID + "-public"
		}
		models = append(models, v1beta1.Model{
			ModelStorage: modelID,
			MountPath:    pm.PretrainedModelPath,
		})

	}

	juypterlab := v1beta1.JuypterLab{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      job.JobName,
			Labels: map[string]string{
				v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
				v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
			},
		},
		Spec: v1beta1.JuypterLabSpec{
			GPUCount:       job.GPUCount,
			Memory:         job.Memory,
			CPUCount:       job.CPUCount,
			GPUProduct:     job.GPUProduct,
			DataVolumeSize: job.DataVolumeSize,
			Models:         models,
		},
	}

	return s.kubeClient.Create(ctx, &juypterlab)
}
