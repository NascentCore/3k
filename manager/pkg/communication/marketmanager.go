package communication

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sxwl/3k/manager/pkg/config"
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
	"sxwl/3k/manager/pkg/resource"
	"time"
)

// NO_TEST_NEEDED

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
	CkptVol     int    `json:"ckptVol"`
	DatasetPath string `json:"datasetPath"`
	GpuNumber   int    `json:"gpuNumber"`
	GpuType     string `json:"gpuType"`
	HfURL       string `json:"hfUrl"`
	ImagePath   string `json:"imagePath"`
	JobID       int    `json:"jobId"`
	JobName     string `json:"jobName"`
	JobType     string `json:"jobType"`
	ModelPath   string `json:"modelPath"`
	ModelVol    int    `json:"modelVol"`
	StopTime    int    `json:"stopTime"`
	StopType    int    `json:"stopType"`
}

type RawJobsData []RawJobDataItem

// communicate with market manager get jobs should run in this cpod.
// return jobs and err , if err != nil , jobs means nothing
func GetJobs(cpodid string) ([]job.Job, error) {
	params := url.Values{}
	jobs := []job.Job{}
	u, err := url.Parse(config.BASE_URL + config.URLPATH_FETCH_JOB)
	if err != nil {
		return jobs, err
	}
	params.Set("cpodid", cpodid)
	u.RawQuery = params.Encode()
	urlPath := u.String()

	req, err := http.NewRequest(http.MethodGet, urlPath, nil)
	if err != nil {
		return jobs, err
	}
	req.Header.Add("Authorization", "Bearer "+config.ACCESS_KEY)
	resp, err := http.DefaultClient.Do(req)
	//resp, err := http.Get(urlPath)
	if err != nil {
		return jobs, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return jobs, errors.New(fmt.Sprintf("status code(%d) not 200", resp.StatusCode))
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return jobs, err
	}
	var parsedData RawJobsData
	err = json.Unmarshal(respData, &parsedData)
	if err != nil {
		return jobs, err
	}
	for _, rawJob := range parsedData {
		jobs = append(jobs, rawJobToJob(rawJob))
	}
	return jobs, nil
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
		CKPTVolumeSize:       rawJob.CkptVol,
		ModelPath:            rawJob.ModelPath,
		ModelVolumeSize:      rawJob.ModelVol,
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
	req, err := http.NewRequest(http.MethodPost, config.BASE_URL+config.URLPATH_UPLOAD_CPOD_STATUS, bytes.NewBuffer(bytesData))
	if err != nil {
		log.SLogger.Errorw("build request error", "error", err)
		return false
	}
	req.Header.Add("Authorization", "Bearer "+config.ACCESS_KEY)
	req.Header.Add("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	//resp, err := http.Post(config.BASE_URL+config.URLPATH_UPLOAD_CPOD_STATUS, "application/json", bytes.NewReader(bytesData))
	if err != nil {
		log.SLogger.Errorw("upload status err", "error", err)
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		respData, err := io.ReadAll(resp.Body)
		if err != nil {
			log.SLogger.Errorw("statuscode != 200 and read body err", "code", resp.StatusCode, "error", err)
		} else {
			log.SLogger.Warnw("statuscode != 200", "code", resp.StatusCode, "body", string(respData))
		}
		return false
	}
	return true
}

// check the given url by make a test call.
func CheckURL(url string) error {
	// TODO: implement it
	return nil
}
