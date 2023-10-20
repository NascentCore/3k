package communication

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/job/state"
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

// communicate with market manager get jobs should run in this cpod.
func GetJobs(cpodid string) []job.Job {
	// TODO: adapt marketmanager interface
	data := map[string]string{
		"cpod_id": cpodid,
	}
	bytesData, err := json.Marshal(data)
	if err != nil {
		return nil
	}
	resp, err := http.Post(baseURL+"/getjobs", "application/json", bytes.NewReader(bytesData))
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil
	}
	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil
	}
	var parsedData map[string]interface{}
	err = json.Unmarshal(respData, &parsedData)
	if err != nil {
		return nil
	}
	// TODO: parsedData -->  job.Job
	return []job.Job{}
}

// upload cpod info to marketmanager.
func UploadCPodStatus(up UploadPayload) bool {
	bytesData, err := json.Marshal(up)
	if err != nil {
		return false
	}
	resp, err := http.Post(baseURL+"/uploadstatus", "application/json", bytes.NewReader(bytesData))
	if err != nil {
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
