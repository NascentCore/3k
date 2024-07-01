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
	"k8s.io/apimachinery/pkg/util/intstr"
	"knative.dev/pkg/ptr"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
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
	storageClassName     string
}

func NewSyncJob(kubeClient client.Client, scheduler sxwl.Scheduler, logger logr.Logger, uploadTrainedModel, autoDownloadReource bool, inferImage, storageClassName string) *SyncJob {
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
		storageClassName:     storageClassName,
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
	readyUsers := s.syncUsers(ctx, users)
	s.logger.Info("initially succeed users", "users", readyUsers)
	s.processTrainningJobs(ctx, readyUsers, portalTrainningJobs)
	s.processFinetune(ctx, readyUsers, portalTrainningJobs)
	s.processInferenceJobs(ctx, readyUsers, portalInferenceJobs)
	s.processJupyterLabJobs(ctx, portalJupyterLabJobs, readyUsers)

}

func (s *SyncJob) syncUsers(ctx context.Context, userIDs []sxwl.UserID) []sxwl.UserID {
	var readyUser []sxwl.UserID
	var users v1.NamespaceList
	err := s.kubeClient.List(ctx, &users, client.HasLabels{
		v1beta1.CPodUserNamespaceLabel,
	})
	if err != nil {
		s.logger.Error(err, "failed to list users")
		return readyUser
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
							Namespace: userNs.Name,
						},
					},
					RoleRef: rbacv1.RoleRef{
						Kind:     "ClusterRole",
						Name:     "cluster-admin",
						APIGroup: "rbac.authorization.k8s.io",
					},
				}
				if err = s.kubeClient.Create(ctx, &newClusterRoleBinding); err != nil {
					s.logger.Error(err, "failed to create clusterrolebinding", "user", userNs.Name, "clusterrolebinding", newClusterRoleBinding)
					continue
				} else {
					s.logger.Info("clusterrolebinding created", "user", userNs.Name)
				}
			} else {
				s.logger.Error(err, "failed to get clusterrolebinding", "user", userNs.Name)
				continue
			}

		}

		var tensorboard appsv1.StatefulSet
		if err := s.kubeClient.Get(ctx, types.NamespacedName{Namespace: userNs.Name, Name: "tensorboard"}, &tensorboard); err != nil {
			if kerrors.IsNotFound(err) {
				// 创建 svc
				if err := s.kubeClient.Create(ctx, &v1.Service{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "tensorboard-svc",
						Namespace: userNs.Name,
					},
					Spec: v1.ServiceSpec{
						Selector: map[string]string{
							"app": "tensorboard",
						},
						Ports: []v1.ServicePort{
							{
								Port:       6006,
								TargetPort: intstr.FromInt32(6006),
							},
						},
					},
				}); err != nil && !kerrors.IsAlreadyExists(err) {
					s.logger.Error(err, "failed to init tensorboard svc", "user", userNs.Name)
					continue
				}
				// 创建 ing

				pathType := networkingv1.PathTypeImplementationSpecific
				if err := s.kubeClient.Create(ctx, &networkingv1.Ingress{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "tensorboard-ing",
						Namespace: userNs.Name,
						Annotations: map[string]string{
							"nginx.ingress.kubernetes.io/rewrite-target": fmt.Sprintf("/tensorboard/%v/$2", userNs.Name),
						},
					},
					Spec: networkingv1.IngressSpec{
						IngressClassName: ptr.String("nginx"),
						Rules: []networkingv1.IngressRule{
							{
								IngressRuleValue: networkingv1.IngressRuleValue{
									HTTP: &networkingv1.HTTPIngressRuleValue{
										Paths: []networkingv1.HTTPIngressPath{
											{
												PathType: &pathType,
												Path:     fmt.Sprintf("/tensorboard/%v(/|$)(.*)", userNs.Name),
												Backend: networkingv1.IngressBackend{
													Service: &networkingv1.IngressServiceBackend{
														Name: "tensorboard-svc",
														Port: networkingv1.ServiceBackendPort{
															Number: 6006,
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
				}); err != nil && !kerrors.IsAlreadyExists(err) {
					s.logger.Error(err, "failed to init tensorboard svc", "user", userNs.Name)
					continue
				}

				if err := s.kubeClient.Create(ctx, &v1.PersistentVolumeClaim{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: userNs.Name,
						Name:      "log-volume-tensorboard-0",
					},
					Spec: v1.PersistentVolumeClaimSpec{
						AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
						StorageClassName: &s.storageClassName,
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceStorage: resource.MustParse("10Gi"),
							},
						},
					},
				}); err != nil && !kerrors.IsAlreadyExists(err) {
					s.logger.Error(err, "failed to init tensorboard pvc", "user", userNs.Name)
					continue
				}
				// 创建 deploy
				if err := s.kubeClient.Create(ctx, &appsv1.StatefulSet{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: userNs.Name,
						Name:      "tensorboard",
					},
					Spec: appsv1.StatefulSetSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": "tensorboard",
							},
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									"app": "tensorboard",
								},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:  "tensorboard",
										Image: "dockerhub.kubekey.local/kubesphereio/tensorflow:latest",
										Command: []string{
											"tensorboard", "--logdir", "/logs", "--host", "0.0.0.0", "--path_prefix", fmt.Sprintf("/tensorboard/%v/", userNs.Name),
										},
										VolumeMounts: []v1.VolumeMount{
											{
												Name:      "log-volume",
												MountPath: "/logs",
											},
										},
									},
								},
								Volumes: []v1.Volume{
									{
										Name: "log-volume",
										VolumeSource: v1.VolumeSource{
											PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
												ClaimName: "log-volume-tensorboard-0",
											},
										},
									},
								},
							},
						},
					},
				}); err != nil && !kerrors.IsAlreadyExists(err) {
					s.logger.Error(err, "failed to init tensorboard statefulset", "user", userNs.Name)
					continue
				}
			} else {
				s.logger.Error(err, "failed to get sts ", "user", userNs.Name)
				continue
			}
		}
		readyUser = append(readyUser, sxwl.UserID(userNs.Name))
	}
	return readyUser
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
			autoMerge := true
			if job.ModelSavedType == "lora" {
				autoMerge = false
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
						Model:           job.PretrainModelName,
						DatasetStorage:  job.DatasetId,
						DatasetIsPublic: job.DatasetIsPublic,
						Upload:          s.uploadTrainedModel,
						HyperParameters: map[string]string{
							"n_epochs":                 job.Epochs,
							"learning_rate_multiplier": job.LearningRate,
							"batch_size":               job.BatchSize,
						},
						GPUCount:   int32(job.GpuNumber),
						GPUProduct: job.GpuType,
						AutoMerge:  autoMerge,
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
		var currentJupyterLabJobs v1beta1.JupyterLabList
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
			var currentJob *v1beta1.JupyterLab
			for i, jl := range currentJupyterLabJobs.Items {
				if jl.Name == job.JobName {
					found = true
					currentJob = &currentJupyterLabJobs.Items[i]
				}
			}

			if !found {
				// 创建新的JupyterLab任务
				err := s.createJupyterLab(ctx, string(UserID), job)
				if err != nil {
					s.logger.Error(err, "failed to create JupyterLab")
				}
			} else {
				currnentRep := int32(0)
				if currentJob.Spec.Replicas == nil {
					currnentRep = 1
				} else {
					currnentRep = *currentJob.Spec.Replicas
				}

				if job.Replicas != currnentRep {
					if err := s.updateJupyterLabReplicas(ctx, currentJob, int(job.Replicas)); err != nil {
						s.logger.Error(err, "failed to update  JupyterLab replicas")
					}
				}

			}
		}

	}
	var totalJuypterlabJobs v1beta1.JupyterLabList
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

func (s *SyncJob) updateJupyterLabReplicas(ctx context.Context, jupyterlab *v1beta1.JupyterLab, replicas int) error {
	jupyterlab.Spec.Replicas = ptr.Int32(int32(replicas))
	return s.kubeClient.Update(ctx, jupyterlab)
}

func (s *SyncJob) createJupyterLab(ctx context.Context, namespace string, job sxwl.PortalJupyterLabJob) error {
	// 处理预训练模型的挂载点
	models := []v1beta1.Model{}
	for _, model := range job.Resource.Models {
		models = append(models, v1beta1.Model{
			Name:          model.ModelName,
			ModelStorage:  model.ModelID,
			ModelIspublic: model.ModelIsPublic,
			ModelSize:     model.ModelSize,
			MountPath:     "",
		})
	}

	for _, adapter := range job.Resource.Adapters {
		models = append(models, v1beta1.Model{
			IsAdapter:     true,
			Name:          adapter.AdapterName,
			ModelStorage:  adapter.AdapterID,
			ModelIspublic: adapter.AdapterIsPublic,
			ModelSize:     adapter.AdapterSize,
			MountPath:     "",
		})

	}

	datasets := []v1beta1.Dataset{}
	for _, dataset := range job.Resource.Datasets {
		datasets = append(datasets, v1beta1.Dataset{
			Name:            dataset.DatasetName,
			DatasetStorage:  dataset.DatasetID,
			DatasetIspublic: dataset.DatasetIsPublic,
			DatasetSize:     dataset.DatasetSize,
			MountPath:       "",
		})

	}

	juypterlab := v1beta1.JupyterLab{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      job.JobName,
			Labels: map[string]string{
				v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
				v1beta1.CPodUserIDLabel:    fmt.Sprint(job.UserID),
			},
		},
		Spec: v1beta1.JupyterLabSpec{
			Replicas:       ptr.Int32(1),
			GPUCount:       job.GPUCount,
			Memory:         job.Memory,
			CPUCount:       job.CPUCount,
			GPUProduct:     job.GPUProduct,
			DataVolumeSize: job.DataVolumeSize,
			Models:         models,
			Datasets:       datasets,
		},
	}

	return s.kubeClient.Create(ctx, &juypterlab)
}
