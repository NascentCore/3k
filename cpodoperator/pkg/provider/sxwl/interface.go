package sxwl

import (
	"net/http"
	"time"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/resource"
)

type PortalTrainningJob struct {
	CkptPath    string            `json:"ckptPath"`
	CkptVol     int               `json:"ckptVol"`
	Command     string            `json:"runCommand"`
	Envs        map[string]string `json:"env"`
	DatasetPath string            `json:"datasetPath"`
	DatasetId   string            `json:"DatasetId"`
	GpuNumber   int               `json:"gpuNumber"`
	GpuType     string            `json:"gpuType"`
	HfURL       string            `json:"hfUrl"`
	// TODO: @sxwl-donggang rename to Image
	ImagePath         string `json:"imagePath"`
	JobID             int    `json:"jobId"`
	JobName           string `json:"jobName"`
	JobType           string `json:"jobType"`
	ModelPath         string `json:"modelPath"`
	ModelVol          int    `json:"modelVol"`
	PretrainModelId   string `json:"pretrainedModelId"`
	PretrainModelPath string `json:"pretrainedModelPath"`
	StopTime          int    `json:"stopTime"`
	StopType          int    `json:"stopType"`
}

type PortalInferenceJob struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
	ModelId     string `json:"model_id"`
	CpodId      string `json:"cpod_id"`
}

type TrainningJobState struct {
	Name      string          `json:"name"`
	Namespace string          `json:"namespace"`
	JobType   v1beta1.JobType `json:"jobtype"`
	// TODO: @sxwl-donggang 序列化风格没保持一致，第一版竟然让sxwl不变更
	JobStatus v1beta1.JobConditionType `json:"job_status"`
	Info      string                   `json:"info,omitempty"` // more info about jobstatus , especially when error occured
	Extension interface{}              `json:"extension"`
}

type Scheduler interface {
	// GetAssignedJobList get assigned to this  Job  from scheduler
	GetAssignedJobList() ([]PortalTrainningJob, []PortalInferenceJob, error)

	// upload heartbeat info ,
	HeartBeat(HeartBeatPayload) error
}

type InferenceJobState struct {
	ServiceName string `json:"service_name"`
	Status      string `json:"status"`
}

type HeartBeatPayload struct {
	CPodID              string                    `json:"cpod_id"`
	TrainningJobsStatus []TrainningJobState       `json:"job_status"`
	InferenceJobsStatus []InferenceJobState       `json:"inference_status"`
	ResourceInfo        resource.CPodResourceInfo `json:"resource_info"`
	UpdateTime          time.Time                 `json:"update_time"`
}

func NewScheduler(baseURL, accesskey, identify string) Scheduler {
	return &sxwl{
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		baseURL:   baseURL,
		accessKey: accesskey,
		identity:  identify,
	}
}
