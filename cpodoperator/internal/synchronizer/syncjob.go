package synchronizer

import (
	"context"
	"strings"
	"sync"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type jobBuffer struct {
	m  map[string]sxwl.PortalTrainningJob
	mu *sync.RWMutex
}

type SyncJob struct {
	kubeClient       client.Client
	scheduler        sxwl.Scheduler
	createFailedJobs jobBuffer
	logger           logr.Logger
}

func NewSyncJob(kubeClient client.Client, scheduler sxwl.Scheduler, logger logr.Logger) *SyncJob {
	return &SyncJob{
		kubeClient:       kubeClient,
		scheduler:        scheduler,
		createFailedJobs: jobBuffer{m: map[string]sxwl.PortalTrainningJob{}, mu: new(sync.RWMutex)},
		logger:           logger,
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

	var cpodjobs v1beta1.CPodJobList
	err := s.kubeClient.List(ctx, &cpodjobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		s.logger.Error(err, "failed to list cpodjob")
		return
	}

	portaljobs, _, err := s.scheduler.GetAssignedJobList()
	if err != nil {
		s.logger.Error(err, "failed to list job")
		return
	}
	s.logger.Info("assigned job", "jobs", portaljobs)

	for _, job := range portaljobs {
		exists := false
		for _, cpodjob := range cpodjobs.Items {
			if cpodjob.Name == job.JobName {
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
			//
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
				s.addCreateFailedJob(job)
				continue
			}
			var backoffLimit int32 = int32(job.BackoffLimit)

			newCPodJob := v1beta1.CPodJob{
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
			if err = s.kubeClient.Create(ctx, &newCPodJob); err != nil {
				s.addCreateFailedJob(job)
				s.logger.Error(err, "failed to create cpodjob", "job", newCPodJob)
			} else {
				s.deleteCreateFailedJob(job.JobName)
				s.logger.Info("cpodjob created", "job", newCPodJob)
			}
		}
	}

	for _, cpodjob := range cpodjobs.Items {
		// do nothing if job has reached a no more change status
		status, _ := parseStatus(cpodjob.Status)
		if status == v1beta1.JobFailed || status == v1beta1.JobModelUploaded ||
			status == v1beta1.JobSucceeded || status == v1beta1.JobModelUploading {
			continue
		}
		exists := false
		for _, job := range portaljobs {
			if cpodjob.Name == job.JobName {
				exists = true
			}
		}
		if !exists {
			if err = s.kubeClient.Delete(ctx, &cpodjob); err != nil {
				s.logger.Error(err, "failed to delete cpodjob")
				return
			}
			s.logger.Info("cpodjob deleted", "jobid", cpodjob.Name)
		}
	}

}

func (s *SyncJob) getCreateFailedJobs() []sxwl.PortalTrainningJob {
	res := []sxwl.PortalTrainningJob{}
	s.createFailedJobs.mu.RLock()
	defer s.createFailedJobs.mu.RUnlock()
	for _, v := range s.createFailedJobs.m {
		res = append(res, v)
	}
	return res
}

func (s *SyncJob) addCreateFailedJob(j sxwl.PortalTrainningJob) {
	if _, ok := s.createFailedJobs.m[j.JobName]; ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	s.createFailedJobs.m[j.JobName] = j
}

// 如果任务创建成功了，将其从失败任务列表中删除
func (s *SyncJob) deleteCreateFailedJob(j string) {
	if _, ok := s.createFailedJobs.m[j]; !ok {
		return
	}
	s.createFailedJobs.mu.Lock()
	defer s.createFailedJobs.mu.Unlock()
	delete(s.createFailedJobs.m, j)
}
