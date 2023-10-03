package communication

import (
	"sxwl/3k/manager/pkg/job"
	"sxwl/3k/manager/pkg/resource"
	"time"
)

var baseURL string // nolint:unused

func SetBaseURL(url string) {
	baseURL = url
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

// communicate with market manager get jobs should run in this cpod.
func GetJobs(cpodid string) []job.Job {
	// TODO: implement it
	return []job.Job{}
}

// upload cpod info to marketmanager.
func UploadCPodStatus(up UploadPayload) bool {
	// TODO: implement it
	return true
}

// check the given url by make a test call.
func CheckURL(url string) error {
	// TODO: implement it
	return nil
}
