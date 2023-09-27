package communication

import (
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/resource"
	"time"
)

var base_url string

func SetBaseURL(url string) {
	base_url = url
}

type UploadPayload struct {
	CPodID       string
	JobStatus    []job.JobStatus
	ResourceDesc resource.ResourceDesc
	UpdateTime   time.Time
}

type JobPayload struct {
	Jobs []job.Job
}

// communicate with market manager get jobs should run in this cpod
func GetJobs(cpodid string) []job.Job {
	//TODO
	return []job.Job{}
}

func UploadCPodStatus(up UploadPayload) bool {
	//TODO
	return true
}

func CheckURL(url string) error {
	//TODO
	return nil
}
