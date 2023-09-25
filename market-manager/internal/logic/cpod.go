package logic

import (
	"encoding/json"
	"github.com/golang/glog"
	"sxwl/3k/mm/internal/handler"
	"sxwl/3k/mm/pkg/db"
)

func CpodJobLogical(body []byte) (string, error) {
	var job db.JobScheduler
	err := json.Unmarshal(body, &job)
	if err != nil {
		glog.Errorf("cpodJobLogical failed to unmarshal job: %v", err)
		return "cpodJobLogical failed to unmarshal job error", err
	}
	return job.CreateJob(handler.DbClinet)
}

func CpodJobResultLogical(body []byte) (string, error) {
	var job db.JobScheduler
	err := json.Unmarshal(body, &job)
	if err != nil {
		glog.Errorf("cpodJobResultLogical failed to unmarshal job result: %v", err)
		return "cpodJobResultLogical failed to unmarshal job result error", err
	}
	return job.UpdateJob(handler.DbClinet)
}

func CpodResourceLogical(body []byte) (string, error) {
	var resources db.CpodResources
	err := json.Unmarshal(body, &resources)
	if err != nil {
		glog.Errorf("cpodResourceLogical failed to unmarshal cpod resource: %v", err)
		return "cpodResourceLogical failed to unmarshal cpod error", err
	}
	for _, resource := range resources.Cpods {
		_, err := resource.CreateResource(handler.DbClinet)
		if err != nil {
			glog.Errorf("cpodResourceLogical failed to create cpod resource: %v", err)
			continue
		}
	}
	return "", nil
}
