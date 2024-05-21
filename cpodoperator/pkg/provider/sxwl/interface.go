package sxwl

import (
	"crypto/tls"
	"net/http"
	"time"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/resource"
)

type PortalJupyterLabJob struct {
	InstanceName string `json:"instanceName"`
	JobName      string `json:"jobName"`
	CPUCount     string `json:"cpuCount"`
	// MIB
	Memory     string `json:"memory"`
	GPUCount   int    `json:"gpuCount"`
	GPUProduct string `json:"gpuProduct"`
	// MIB
	DataVolumeSize   string              `json:"dataVolumeSize"`
	PretrainedModels *[]PretrainedModels `json:"pretrainedModels"`
	UserID           int64               `json:"userId"`
}

type PretrainedModels struct {
	PretrainedModelId       string `json:"pretrainedModelId"`
	PretrainedModelName     string `json:"pretrainedModelName"`
	PretrainedModelPath     string `json:"pretrainedModelPath"`
	PretrainedModelIsPublic bool   `json:"pretrainedModelIsPublic"`
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
	ImagePath             string `json:"imagePath"`
	JobID                 int    `json:"jobId"`
	JobName               string `json:"jobName"`
	JobType               string `json:"jobType"`
	ModelPath             string `json:"modelPath"`
	ModelVol              int    `json:"modelVol"`
	PretrainModelId       string `json:"pretrainedModelId"`
	PretrainModelName     string `json:"pretrainedModelName"`
	PretrainModelUrl      string `json:"pretrainedModelUrl"`
	PretrainModelSize     int    `json:"pretrainedModelSize"`
	PretrainModelPath     string `json:"pretrainedModelPath"`
	PretrainModelIsPublic bool   `json:"pretrainModelIsPublic"`
	StopTime              int    `json:"stopTime"`
	StopType              int    `json:"stopType"`
	BackoffLimit          int    `json:"backoffLimit"`
	Epochs                string `json:"epochs"`
	LearningRate          string `json:"learningRate"`
	BatchSize             string `json:"batchSize"`
	UserID                int64  `json:"userId"`
	TrainedModelName      string `json:"trainedModelName,optional"`
}

type PortalInferenceJob struct {
	ServiceName   string `json:"service_name"`
	Status        string `json:"status"`
	ModelId       string `json:"model_id"`
	ModelIsPublic bool   `json:"model_is_public"`
	GpuType       string `json:"gpu_type"`
	GpuNumber     int64  `json:"gpu_number"`
	CpodId        string `json:"cpod_id"`
	Template      string `json:"template,omitempty"`
	UserID        int64  `json:"user_id"`
}

type TrainningJobState struct {
	Name      string          `json:"name"`
	Namespace string          `json:"namespace"`
	JobType   v1beta1.JobType `json:"jobtype"`
	// TODO: @sxwl-donggang 序列化风格没保持一致，第一版竟然让sxwl不变更
	JobStatus v1beta1.JobConditionType `json:"job_status"`
	Info      string                   `json:"info,omitempty"` // more info about jobstatus , especially when error occured
	Extension interface{}              `json:"extension"`
	UserID    int64                    `json:"user_id"`
}

type UserID string // 用户ID

type Scheduler interface {
	// GetAssignedJobList get assigned to this  Job  from scheduler
	GetAssignedJobList() ([]PortalTrainningJob, []PortalInferenceJob, []PortalJupyterLabJob, []UserID, error)

	// upload heartbeat info ,
	HeartBeat(HeartBeatPayload) error
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

type HeartBeatPayload struct {
	CPodID               string                    `json:"cpod_id"`
	TrainningJobsStatus  []TrainningJobState       `json:"job_status"`
	InferenceJobsStatus  []InferenceJobState       `json:"inference_status"`
	JupyterLabJobsStatus []JupyterLabJobState      `json:"jupyter_status"`
	ResourceInfo         resource.CPodResourceInfo `json:"resource_info"`
	UpdateTime           time.Time                 `json:"update_time"`
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
