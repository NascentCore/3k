package synchronizer

import (
	"context"
	"strconv"
	"strings"

	"github.com/NascentCore/cpodoperator/api/v1beta1"

	"time"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"
	"github.com/NascentCore/cpodoperator/pkg/resource"
	"github.com/NascentCore/cpodoperator/pkg/util"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	InferenceJobDataPreparing = "dataPreparing"
	InferenceJobDeploying     = "pending"
	InferenceJobDeployed      = "running"
	InferenceJobFailed        = "failed"
)

type CPodObserver struct {
	kubeClient    client.Client
	logger        logr.Logger
	ch            chan<- sxwl.HeartBeatPayload
	cpodId        string
	cpodNamespace string
}

func NewCPodObserver(kubeClient client.Client, cpodId, cpodNamespace string, ch chan<- sxwl.HeartBeatPayload, logger logr.Logger) *CPodObserver {
	return &CPodObserver{kubeClient: kubeClient, logger: logger, ch: ch,
		cpodId: cpodId, cpodNamespace: cpodNamespace}
}

func (co *CPodObserver) Start(ctx context.Context) {
	co.logger.Info("cpod observer")
	js, err := co.getTrainningJobStates(ctx)
	if err != nil {
		co.logger.Error(err, "get job state error")
		return
	}
	// combine with finetune jobs
	fs, err := co.getFinetuneStates(ctx)
	if err != nil {
		co.logger.Error(err, "get finetune job state error")
		return
	}
	js = append(js, fs...)

	co.logger.Info("jobstates to upload", "js", js)
	// inferencejobs
	ijs, err := co.getInferenceJobStates(ctx)
	if err != nil {
		co.logger.Error(err, "get inference job state error")
		return
	}
	co.logger.Info("inference jobstates to upload", "ijs", ijs)
	resourceInfo, err := co.getResourceInfo(ctx)
	if err != nil {
		co.logger.Error(err, "get resource error")
		return
	}
	// jupyterlab jobs
	jjs, err := co.getJupyterLabJobStates(ctx)
	if err != nil {
		co.logger.Error(err, "get jupyterlab job state error")
		return
	}
	co.ch <- sxwl.HeartBeatPayload{
		CPodID:               co.cpodId,
		ResourceInfo:         resourceInfo,
		TrainningJobsStatus:  js,
		InferenceJobsStatus:  ijs,
		JupyterLabJobsStatus: jjs,
		UpdateTime:           time.Now(),
	}
	co.logger.Info("upload payload refreshed")
}

func parseStatus(cpodjob v1beta1.CPodJob) v1beta1.Phase {
	if util.IsFinshed(cpodjob.Status) {
		if util.IsSucceeded(cpodjob.Status) && (!cpodjob.Spec.UploadModel) || (cpodjob.Spec.UploadModel && util.IsModelUploaded(cpodjob.Status)) {
			return v1beta1.PhaseSucceeded
		} else if util.IsFailed(cpodjob.Status) {
			return v1beta1.PhaseFailed
		} else {
			return v1beta1.PhaseRunning
		}
	}
	if cond := util.GetCondition(cpodjob.Status, v1beta1.JobDataPreparing); cond != nil {
		if cond.Status == corev1.ConditionFalse {
			return v1beta1.PhasePreparingData
		}
	}
	return v1beta1.PhaseRunning
}

func (co *CPodObserver) getTrainningJobStates(ctx context.Context) ([]sxwl.TrainningJobState, error) {
	var cpodjobs v1beta1.CPodJobList
	err := co.kubeClient.List(ctx, &cpodjobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		return nil, err
	}

	stats := []sxwl.TrainningJobState{}
	for _, cpodjob := range cpodjobs.Items {
		status := parseStatus(cpodjob)
		// it'a time limit job
		Name := cpodjob.Name
		JobType := cpodjob.Spec.JobType

		for _, owner := range cpodjob.OwnerReferences {
			if owner.Kind == "FineTune" {
				JobType = "Finetune"
				Name = owner.Name
				break
			}
		}

		stats = append(stats, sxwl.TrainningJobState{
			Name:      Name,
			Namespace: cpodjob.Namespace,
			JobType:   JobType,
			// TODO: synch defination with portal
			JobStatus: v1beta1.JobConditionType(strings.ToLower(string(status))),
		})
	}
	return stats, nil
}

func (co *CPodObserver) getJupyterLabJobStates(ctx context.Context) ([]sxwl.JupyterLabJobState, error) {
	var jupyterlabs v1beta1.JupyterLabList
	states := []sxwl.JupyterLabJobState{}
	err := co.kubeClient.List(ctx, &jupyterlabs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		return states, err
	}

	for _, ss := range jupyterlabs.Items {

		status := string(ss.Status.Phase)
		if ss.Status.Phase == v1beta1.JupyterLabJobPhaseRunning {
			status = string(v1beta1.JupyterLabJobPhaseRunning)
		} else if ss.Status.DataReady != true {
			status = "DataPreparing"
		}

		state := sxwl.JupyterLabJobState{
			JobName: ss.Name,
			Status:  strings.ToLower(status),
			URL:     "/jupyterlab/" + ss.Name,
		}
		states = append(states, state)
	}
	return states, nil
}

func (co *CPodObserver) getFinetuneStates(ctx context.Context) ([]sxwl.TrainningJobState, error) {
	var finetunes v1beta1.FineTuneList
	err := co.kubeClient.List(ctx, &finetunes, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		co.logger.Error(err, "failed to list finetunejob")
	}

	var stats []sxwl.TrainningJobState
	for _, finetuneJob := range finetunes.Items {
		stats = append(stats, sxwl.TrainningJobState{
			Name:      finetuneJob.Name,
			Namespace: finetuneJob.Namespace,
			JobType:   "Finetune",
			JobStatus: v1beta1.JobConditionType(strings.ToLower(string(finetuneJob.Status.Phase))),
			Info:      finetuneJob.Status.FailureMessage,
		})
	}
	return stats, nil
}

func (co *CPodObserver) getInferenceJobStates(ctx context.Context) ([]sxwl.InferenceJobState, error) {
	var inferenceJobs v1beta1.InferenceList
	err := co.kubeClient.List(ctx, &inferenceJobs, &client.MatchingLabels{
		v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource,
	})
	if err != nil {
		return nil, err
	}

	stats := []sxwl.InferenceJobState{}
	for _, inferenceJob := range inferenceJobs.Items {
		status := InferenceJobDeploying
		if inferenceJob.Status.Ready {
			status = InferenceJobDeployed
		} else if !inferenceJob.Status.DataReady {
			status = InferenceJobDataPreparing
		}
		url := "/inference/" + inferenceJob.Name

		stats = append(stats, sxwl.InferenceJobState{
			ServiceName: inferenceJob.Name,
			Status:      strings.ToLower(status),
			URL:         url,
		})
	}
	return stats, nil
}

func (co *CPodObserver) getResourceInfo(ctx context.Context) (resource.CPodResourceInfo, error) {
	var info resource.CPodResourceInfo
	info.CPodID = co.cpodId
	info.CPodVersion = "v1.0"
	// get node list from k8s
	info.Nodes = []resource.NodeInfo{}
	var nodeInfo corev1.NodeList
	err := co.kubeClient.List(ctx, &nodeInfo)
	if err != nil {
		return info, err
	}
	for _, node := range nodeInfo.Items {
		t := resource.NodeInfo{}
		t.Name = node.Name
		t.Status = node.Labels["status"]
		t.KernelVersion = node.Labels["feature.node.kubernetes.io/kernel-version.full"]
		t.LinuxDist = node.Status.NodeInfo.OSImage
		t.Arch = node.Labels["kubernetes.io/arch"]
		t.CPUInfo.Cores = int(node.Status.Capacity.Cpu().Value())
		if v, ok := node.Labels[v1beta1.K8S_LABEL_NV_GPU_PRODUCT]; ok {
			t.GPUInfo.Prod = v
			t.GPUInfo.Vendor = "nvidia"
			t.GPUInfo.Driver = node.Labels["nvidia.com/cuda.driver.major"] + "." +
				node.Labels["nvidia.com/cuda.driver.minor"] + "." +
				node.Labels["nvidia.com/cuda.driver.rev"]
			t.GPUInfo.CUDA = node.Labels["nvidia.com/cuda.runtime.major"] + "." +
				node.Labels["nvidia.com/cuda.runtime.minor"]
			t.GPUInfo.MemSize, _ = strconv.Atoi(node.Labels["nvidia.com/gpu.memory"])
			t.GPUInfo.Status = "abnormal"
			if node.Labels[v1beta1.K8S_LABEL_NV_GPU_PRESENT] == "true" {
				t.GPUInfo.Status = "normal"
			}
			// init GPUState Array , accordding to nvidia.com/gpu.count label
			t.GPUState = []resource.GPUState{}
			gpuCnt, _ := strconv.Atoi(node.Labels["nvidia.com/gpu.count"])
			t.GPUTotal = gpuCnt
			for i := 0; i < gpuCnt; i++ {
				t.GPUState = append(t.GPUState, resource.GPUState{})
			}
		}
		t.MemInfo.Size = int(node.Status.Allocatable.Memory().Value() / 1024 / 1024)
		info.Nodes = append(info.Nodes, t)
	}

	pods := &corev1.PodList{}
	if err := co.kubeClient.List(ctx, pods); err != nil {
		return info, err
	}
	for _, pod := range pods.Items {
		for _, container := range pod.Spec.Containers {
			if gpu, ok := container.Resources.Requests["nvidia.com/gpu"]; ok {
				for i, node := range info.Nodes {
					if node.Name == pod.Spec.NodeName {
						co.logger.Info("gpu used", "gpu", gpu.Value(), "pod", pod.Name, "podNamespace", pod.Namespace)
						info.Nodes[i].GPUUsed += int(gpu.Value())
						info.Nodes[i].MemInfo.Used += int(container.Resources.Requests.Memory().Value() / 1024 / 1024)
						info.Nodes[i].CPUInfo.Used += int(container.Resources.Requests.Cpu().Value())
					}
				}
			}
		}
	}

	for i, _ := range info.Nodes {
		info.Nodes[i].GPUAllocatable = info.Nodes[i].GPUTotal - info.Nodes[i].GPUUsed
	}

	// stat gpus in cpod
	statTotal := map[[2]string]int{}
	statAlloc := map[[2]string]int{}
	statMemSize := map[[2]string]int{}
	for _, node := range info.Nodes {
		statTotal[[2]string{node.GPUInfo.Vendor, node.GPUInfo.Prod}] += node.GPUTotal
		statAlloc[[2]string{node.GPUInfo.Vendor, node.GPUInfo.Prod}] += node.GPUAllocatable
		statMemSize[[2]string{node.GPUInfo.Vendor, node.GPUInfo.Prod}] = node.GPUInfo.MemSize
	}
	for k, v := range statTotal {
		info.GPUSummaries = append(info.GPUSummaries, resource.GPUSummary{
			Vendor:      k[0],
			Prod:        k[1],
			MemSize:     statMemSize[k],
			Total:       v,
			Allocatable: statAlloc[k],
		})
	}

	caches, err := co.getExistingArtifacts(ctx)
	if err != nil {
		return resource.CPodResourceInfo{}, err
	}
	info.Caches = caches
	return info, nil
}

// return dataset list \ model list \ err
func (co *CPodObserver) getExistingArtifacts(ctx context.Context) ([]resource.Cache, error) {
	var modelList cpodv1.ModelStorageList
	err := co.kubeClient.List(ctx, &modelList)
	if err != nil {
		return nil, err
	}
	caches := []resource.Cache{}
	for _, model := range modelList.Items {
		FinetuneGPUCount := 1
		InferenceGPUCount := 1

		isPublic := false
		if model.Namespace == "public" {
			isPublic = true
		}
		if model.Labels != nil {
			if _, ok := model.Labels[v1beta1.CPODStorageCopyLable]; ok {
				continue
			}
			if value, ok := model.Labels[v1beta1.CPodModelstorageDefaultFinetuneGPUCount]; ok {
				FinetuneGPUCount, _ = strconv.Atoi(value)
			}

			if value, ok := model.Labels[v1beta1.CPodModelstorageDefaultInferenceGPUCount]; ok {
				InferenceGPUCount, _ = strconv.Atoi(value)
			}
		}

		if model.Spec.IsLoraAdapter {
			caches = append(caches, resource.Cache{
				IsPublic:          isPublic,
				UserID:            model.Namespace,
				DataType:          resource.CacheAdapter,
				DataName:          model.Spec.ModelName,
				DataId:            model.Name,
				DataSize:          model.Status.Size,
				Template:          model.Spec.Template,
				DataSource:        model.Spec.ModelType,
				FinetuneGPUCount:  int64(FinetuneGPUCount),
				InferenceGPUCount: int64(InferenceGPUCount),
			})
		} else {
			caches = append(caches, resource.Cache{
				IsPublic:          isPublic,
				UserID:            model.Namespace,
				DataType:          resource.CacheModel,
				DataName:          model.Spec.ModelName,
				DataId:            model.Name,
				DataSize:          model.Status.Size,
				Template:          model.Spec.Template,
				DataSource:        model.Spec.ModelType,
				FinetuneGPUCount:  int64(FinetuneGPUCount),
				InferenceGPUCount: int64(InferenceGPUCount),
			})

		}

	}
	var datasetList cpodv1.DataSetStorageList
	err = co.kubeClient.List(ctx, &datasetList)
	if err != nil {
		return nil, err
	}
	for _, dataset := range datasetList.Items {
		if dataset.Labels != nil {
			if _, ok := dataset.Labels[v1beta1.CPODStorageCopyLable]; ok {
				continue
			}
		}
		isPublic := false
		if dataset.Namespace == "public" {
			isPublic = true
		}

		caches = append(caches, resource.Cache{
			IsPublic:   isPublic,
			UserID:     dataset.Namespace,
			DataType:   resource.CacheDataSet,
			DataName:   dataset.Spec.DatasetName,
			DataId:     dataset.Name,
			DataSize:   dataset.Status.Size,
			DataSource: dataset.Spec.DatasetType,
		})
	}
	return caches, nil
}

// TODO: read image list from harbor
func (co *CPodObserver) getImages(ctx context.Context) ([]string, error) {
	return []string{}, nil
}
