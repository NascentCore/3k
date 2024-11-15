package sxwl

import (
	"crypto/tls"
	"net/http"
	"time"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/resource"
)

type PortalYAMLResource struct {
	JobName string `json:"job_name"`
	YAML    string `json:"crd"`
	AppName string `json:"app_name"`
	AppID   string `json:"app_id"`
	UserID  string `json:"user_id"`
	Meta    string `json:"meta"`
}

type PortalJupyterLabJob struct {
	InstanceName string `json:"instanceName"`
	JobName      string `json:"jobName"`
	CPUCount     string `json:"cpuCount"`
	// MIB
	Memory     string `json:"memory"`
	GPUCount   int    `json:"gpuCount"`
	GPUProduct string `json:"gpuProduct"`
	// MIB
	DataVolumeSize string              `json:"dataVolumeSize"`
	Resource       *JupyterLabResource `json:"resource"`
	UserID         string              `json:"userId"`
	Replicas       int32               `json:"replicas"`
}

type JupyterLabResource struct {
	Models   []Model   `json:"models"`
	Datasets []Dataset `json:"datasets"`
	Adapters []Adapter `json:"adapters"`
}

type Dataset struct {
	DatasetID       string `json:"dataset_id"`
	DatasetName     string `json:"dataset_name"`
	DatasetSize     int    `json:"dataset_size"`
	DatasetPath     string `json:"dataset_path"`
	DatasetIsPublic bool   `json:"dataset_is_public"`
}
type Model struct {
	ModelID       string `json:"model_id"`
	ModelName     string `json:"model_name"`
	ModelSize     int    `json:"model_size"`
	ModelPath     string `json:"model_path"`
	ModelTemplate string `json:"model_template"`
	ModelIsPublic bool   `json:"model_is_public"`
	ModelCategory string `json:"model_category"`
}

type Adapter struct {
	AdapterID       string `json:"adapter_id"`
	AdapterName     string `json:"adapter_name"`
	AdapterSize     int    `json:"adapter_size"`
	AdapterPath     string `json:"adapter_path"`
	AdapterIsPublic bool   `json:"adapter_is_public"`
}

type PortalTrainningJob struct {
	CkptPath        string            `json:"ckptPath"`
	CkptVol         int               `json:"ckptVol"`
	Command         string            `json:"runCommand"`
	Envs            map[string]string `json:"env"`
	DatasetPath     string            `json:"datasetPath"`
	DatasetId       string            `json:"datasetId"`
	DatasetName     string            `json:"datasetName"`
	DatasetUrl      string            `json:"datasetUrl"`
	DatasetSize     int               `json:"datasetSize"`
	DatasetIsPublic bool              `json:"datasetIsPublic"`
	GpuNumber       int               `json:"gpuNumber"`
	GpuType         string            `json:"gpuType"`
	HfURL           string            `json:"hfUrl"`
	// TODO: @sxwl-donggang rename to Image
	ImagePath             string               `json:"imagePath"`
	JobID                 int                  `json:"jobId"`
	JobName               string               `json:"jobName"`
	JobType               string               `json:"jobType"`
	ModelPath             string               `json:"modelPath"`
	ModelVol              int                  `json:"modelVol"`
	PretrainModelId       string               `json:"pretrainedModelId"`
	PretrainModelName     string               `json:"pretrainedModelName"`
	PretrainModelUrl      string               `json:"pretrainedModelUrl"`
	PretrainModelSize     int                  `json:"pretrainedModelSize"`
	PretrainModelPath     string               `json:"pretrainedModelPath"`
	PretrainModelIsPublic bool                 `json:"pretrainModelIsPublic"`
	PretrainModelTemplate string               `json:"pretrainModelTemplate"`
	PretrainModelMeta     string               `json:"pretrainModelMeta"`
	StopTime              int                  `json:"stopTime"`
	StopType              int                  `json:"stopType"`
	BackoffLimit          int                  `json:"backoffLimit"`
	Epochs                string               `json:"epochs"`
	LearningRate          string               `json:"learningRate"`
	BatchSize             string               `json:"batchSize"`
	UserID                string               `json:"userId"`
	TrainedModelName      string               `json:"trainedModelName,optional"`
	ModelSavedType        string               `json:"model_saved_type"`
	FinetuneType          v1beta1.FinetuneType `json:"finetuneType"`
}

type PortalInferenceJob struct {
	ServiceName     string `json:"service_name"`
	Status          string `json:"status"`
	ModelId         string `json:"model_id"`
	ModelName       string `json:"model_name"`
	ModelSize       int    `json:"model_size"`
	ModelIsPublic   bool   `json:"model_is_public"`
	ModelMeta       string `json:"model_meta"`
	GpuType         string `json:"gpu_type"`
	GpuNumber       int64  `json:"gpu_number"`
	CpodId          string `json:"cpod_id"`
	Template        string `json:"template,omitempty"`
	UserID          string `json:"user_id"`
	AdapterId       string `json:"adapter_id"`
	AdapterName     string `json:"adapter_name"`
	AdapterIsPublic bool   `json:"adapter_is_public"`
	AdapterSize     int    `json:"adapter_size"`
	ModelCategory   string `json:"model_category"`
	MinInstances    int    `json:"min_instances,omitempty"`
	MaxInstances    int    `json:"max_instances,omitempty"`
}

type TrainningJobState struct {
	Name      string          `json:"name"`
	Namespace string          `json:"namespace"`
	JobType   v1beta1.JobType `json:"jobtype"`
	// TODO: @sxwl-donggang 序列化风格没保持一致，第一版竟然让sxwl不变更
	JobStatus v1beta1.JobConditionType `json:"job_status"`
	Info      string                   `json:"info,omitempty"` // more info about jobstatus , especially when error occured
	Extension interface{}              `json:"extension"`
	UserID    string                   `json:"user_id"`
}

type UserID string // 用户ID

type Scheduler interface {
	// GetAssignedJobList get assigned to this  Job  from scheduler
	GetAssignedJobList() ([]PortalTrainningJob, []PortalInferenceJob, []PortalJupyterLabJob, []PortalYAMLResource, []UserID, error)

	// upload heartbeat info ,
	HeartBeat(HeartBeatPayload) error

	// upload resource info
	UploadResource(resource ResourceInfo) error
}

type InferenceJobState struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
	URL         string `json:"url"`
}

type JupyterLabJobState struct {
	JobName string `json:"job_name"`
	Status  string `json:"status"`
	URL     string `json:"url"`
}

type YAMLResourceState struct {
	JobName string `json:"job_name"`
	Status  string `json:"status"`
	URL     string `json:"url"`
}

type HeartBeatPayload struct {
	CPodID               string                    `json:"cpod_id"`
	TrainningJobsStatus  []TrainningJobState       `json:"job_status"`
	InferenceJobsStatus  []InferenceJobState       `json:"inference_status"`
	JupyterLabJobsStatus []JupyterLabJobState      `json:"jupyter_status"`
	YAMLResourceStatus   []YAMLResourceState       `json:"app_job_status"`
	ResourceInfo         resource.CPodResourceInfo `json:"resource_info"`
	UpdateTime           time.Time                 `json:"update_time"`
}

type ResourceInfo struct {
	ResourceId   string `json:"resource_id"`
	ResourceType string `json:"resource_type"`
	ResourceName string `json:"resource_name"`
	ResourceSize int    `json:"resource_size"`
	IsPublic     bool   `json:"is_public"`
	Meta         string `json:"meta"`
	UserID       string `json:"user_id"`
}

func NewScheduler(baseURL, accesskey, identify string) Scheduler {
	return &sxwl{
		// 临时修改，需要在域名可用时再改回来
		httpClient: &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true, // 注意：这会禁用证书验证，请谨慎使用
				},
			},
		},
		baseURL:   baseURL,
		accessKey: accesskey,
		identity:  identify,
	}
}
