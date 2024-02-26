package synchronizer

import (
	"context"
	"strings"
	"sync"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"
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
	s.processInferenceJobs(ctx, portalInferenceJobs)

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
			//判断指定的预训练模型是否存在
			if job.PretrainModelId != "" {
				exists, err := s.checkModelExistence(v1beta1.CPOD_NAMESPACE, job.PretrainModelId)
				if err != nil {
					s.logger.Error(err, "failed to check model existence")
					return
				}
				if !exists {
					// TODO: create downloader task and return.
					s.logger.Info("model not exists , starting downloader task , task will be started when downloader task finish")
					// return create failed status during the downloader task.
					s.addCreateFailedTrainningJob(job)
					return
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
				s.logger.Info("invalid jobtype", "jobtype", job.JobType)
				s.addCreateFailedTrainningJob(job)
				continue
			}
			var backoffLimit int32 = int32(job.BackoffLimit)

			newJob := v1beta1.CPodJob{
				ObjectMeta: metav1.ObjectMeta{
					// TODO: create namespace for different tenant
					Namespace: v1beta1.CPOD_NAMESPACE,
					Name:      job.JobName,
					Labels:    map[string]string{v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource},
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
			//create
			newJob := v1beta1.Inference{
				ObjectMeta: metav1.ObjectMeta{
					// TODO: create namespace for different tenant
					Namespace: v1beta1.CPOD_NAMESPACE,
					Name:      job.ServiceName,
					Labels:    map[string]string{v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource},
				},
				// TODO: fill PredictorSpec with infomation provided by portaljob
				Spec: v1beta1.InferenceSpec{
					Predictor: kservev1beta1.PredictorSpec{
						PodSpec: kservev1beta1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "kserve-container",
									Image: "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:latest",
									Command: []string{
										"python",
										"src/api_demo.py",
										"--model_name_or_path",
										"/mnt/models",
										"--template",
										"default",
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
func (s *SyncJob) checkModelExistence(namespace, m string) (bool, error) {
	var ms v1beta1.ModelStorage
	err := s.kubeClient.Get(context.TODO(), types.NamespacedName{Namespace: namespace, Name: m}, &ms)
	if err != nil {
		if kerrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}
