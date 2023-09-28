package logic

import (
	"encoding/json"
	"sxwl/mm/pkg/db"

	"github.com/golang/glog"
)

// cpod jobs scheduler logical, the body is jobs message
func CpodJobLogical(body []byte) (string, error) {
	var job db.JobScheduler
	err := json.Unmarshal(body, &job)
	if err != nil {
		glog.Errorf("cpodJobLogical failed to unmarshal job: %v", err)
		return "cpodJobLogical failed to unmarshal job error", err
	}

	return "", nil
}

// Record cpods jobs result, the body is jobs result
func CpodJobResultLogical(body []byte) (string, error) {
	var job db.JobScheduler
	err := json.Unmarshal(body, &job)
	if err != nil {
		glog.Errorf("cpodJobResultLogical failed to unmarshal job result: %v", err)
		return "cpodJobResultLogical failed to unmarshal job result error", err
	}
	return "", nil
}

// Record cpod resource message cotain GPU and so on, body is the resource message
func CpodResourceLogical(body []byte) (string, error) {
	var resources db.CpodResources
	err := json.Unmarshal(body, &resources)
	if err != nil {
		glog.Errorf("cpodResourceLogical failed to unmarshal cpod resource: %v", err)
		return "cpodResourceLogical failed to unmarshal cpod error", err
	}
	return "", nil
}
