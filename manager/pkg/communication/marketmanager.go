package communication

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/resource"
	"time"
)

var baseURL string //nolint:unused

func SetBaseURL(url string) {
	baseURL = url
}

type UploadPayload struct {
	CPodID       string                    `json:"cpod_id"`
	JobStatus    []state.State             `json:"job_status"`
	ResourceInfo resource.CPodResourceInfo `json:"resource_info"`
	UpdateTime   time.Time                 `json:"update_time"`
}

type JobPayload struct {
	Jobs []job.Job `json:"jobs"`
}

type RawJobDataItem struct {
	CkptPath    string `json:"ckptPath"`
	CkptVol     string `json:"ckptVol"`
	DatasetPath string `json:"datasetPath"`
	GpuNumber   int    `json:"gpuNumber"`
	GpuType     string `json:"gpuType"`
	HfURL       string `json:"hfUrl"`
	ImagePath   string `json:"imagePath"`
	JobID       int    `json:"jobId"`
	JobName     string `json:"jobName"`
	JobType     string `json:"jobType"`
	ModelPath   string `json:"modelPath"`
	ModelVol    string `json:"modelVol"`
	StopTime    int    `json:"stopTime"`
	StopType    int    `json:"stopType"`
}

type RawJobsData []RawJobDataItem

// communicate with market manager get jobs should run in this cpod.
func GetJobs(cpodid string) []job.Job {
	params := url.Values{}
	u, err := url.Parse(baseURL + "/api/userJob/getjob")
	if err != nil {
		log.SLogger.Errorw("url error", "error", err)
		return nil
	}
	params.Set("cpodid", cpodid)
	u.RawQuery = params.Encode()
	urlPath := u.String()
	resp, err := http.Get(urlPath)
	if err != nil {
		log.SLogger.Errorw("http get err", "error", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		log.SLogger.Errorw("status code not 200")
		return nil
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		log.SLogger.Errorw("read body err", "error", err)
		return nil
	}
	fmt.Println(string(respData))
	var parsedData RawJobsData
	err = json.Unmarshal(respData, &parsedData)
	if err != nil {
		log.SLogger.Errorw("parse data err", "error", err)
		return nil
	}
	res := []job.Job{}
	for _, rawJob := range parsedData {
		res = append(res, rawJobToJob(rawJob))
	}
	return res
}

func rawJobToJob(rawJob RawJobDataItem) job.Job {
	gpuPerWorker := 8
	replicas := 1
	if rawJob.GpuNumber < 8 {
		gpuPerWorker = rawJob.GpuNumber
	} else {
		replicas = rawJob.GpuNumber / 8
	}

	return job.Job{
		JobID:                rawJob.JobName,
		JobType:              job.Type(rawJob.JobType),
		Image:                rawJob.ImagePath,
		DataPath:             rawJob.DatasetPath,
		CKPTPath:             rawJob.CkptPath,
		ModelPath:            rawJob.ModelPath,
		GPUType:              rawJob.GpuType,
		GPURequiredPerWorker: gpuPerWorker,
		Replicas:             replicas,
		HuggingFaceURL:       rawJob.HfURL,
		Duration:             rawJob.StopTime,
		StopType:             rawJob.StopType,
	}
}

// upload cpod info to marketmanager.
func UploadCPodStatus(up UploadPayload) bool {
	bytesData, err := json.Marshal(up)
	if err != nil {
		log.SLogger.Errorw("data error", "error", err)
		return false
	}
	resp, err := http.Post(baseURL+"/api/userJob/putPodStatus", "application/json", bytes.NewReader(bytesData))
	if err != nil {
		log.SLogger.Errorw("upload status err", "error", err)
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// check the given url by make a test call.
func CheckURL(url string) error {
	// TODO: implement it
	return nil
}
