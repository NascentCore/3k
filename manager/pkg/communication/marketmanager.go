package communication

import (
	"sxwl/3k/manager/pkg/cluster"
	"sxwl/3k/manager/pkg/job"
	"time"
)

type UploadPayload struct {
	CPodID       string
	JobStatus    []job.JobStatus
	ResourceDesc cluster.ResourceDesc
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
