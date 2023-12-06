package kubeflowmpijob

import (
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job/state"
	"testing"
	"time"
)

func getStateData(deadline string, withCreated, withRunning, withSucceed, withFailed bool) map[string]interface{} {
	conds := []interface{}{}
	if withCreated {
		conds = append(conds, map[string]interface{}{
			"lastTransitionTime": "2023-11-15T07:05:51Z",
			"lastUpdateTime":     "2023-11-15T07:05:51Z",
			"message":            "MPIJob cpod/job1 is created.",
			"reason":             "MPIJobCreated",
			"status":             "True",
			"type":               "Created",
		})
	}
	if withRunning {
		conds = append(conds, map[string]interface{}{
			"lastTransitionTime": "2023-11-15T07:06:03Z",
			"lastUpdateTime":     "2023-11-15T07:06:03Z",
			"message":            "MPIJob cpod/job1 is running.",
			"reason":             "MPIJobRunning",
			"status":             "True",
			"type":               "Running",
		})
	}
	if withFailed {
		conds = append(conds, map[string]interface{}{
			"lastTransitionTime": "2023-11-15T07:06:03Z",
			"lastUpdateTime":     "2023-11-15T07:06:03Z",
			"message":            "MPIJob cpod/job1 is running.",
			"reason":             "MPIJobRunning",
			"status":             "True",
			"type":               "Failed",
		})
	}
	if withSucceed {
		conds = append(conds, map[string]interface{}{
			"lastTransitionTime": "2023-11-15T07:06:03Z",
			"lastUpdateTime":     "2023-11-15T07:06:03Z",
			"message":            "MPIJob cpod/job1 is running.",
			"reason":             "MPIJobRunning",
			"status":             "True",
			"type":               "Succeeded",
		})
	}

	return map[string]interface{}{
		"apiVersion": "kubeflow.org/v2beta1",
		"kind":       "MPIJob",
		"metadata": map[string]interface{}{
			"creationTimestamp": "2023-11-15T07:05:51Z",
			"generation":        1,
			"labels": map[string]interface{}{
				"deadline": deadline,
			},
			"name":            "job1",
			"namespace":       "cpod",
			"resourceVersion": "2830601",
			"uid":             "4644d519-b1bb-4a3c-bedc-889ff57115e3",
		},
		"status": map[string]interface{}{
			"conditions": conds,
			"replicaStatuses": map[string]interface{}{
				"Launcher": map[string]interface{}{
					"active": 1,
				},
				"Worker": map[string]interface{}{
					"active": 1,
				},
			},
			"startTime": "2023-11-15T07:05:51Z",
		},
	}
}

func TestParseState(t *testing.T) {
	sSucceed := state.State{
		Name:      "job1",
		Namespace: "cpod",
		JobType:   state.JobTypeMPI,
		JobStatus: state.JobStatusSucceed,
		Extension: nil,
	}
	sRunning := state.State{
		Name:      "job1",
		Namespace: "cpod",
		JobType:   state.JobTypeMPI,
		JobStatus: state.JobStatusRunning,
		Extension: nil,
	}
	sFailed := state.State{
		Name:      "job1",
		Namespace: "cpod",
		JobType:   state.JobTypeMPI,
		JobStatus: state.JobStatusFailed,
		Extension: nil,
	}
	sCreated := state.State{
		Name:      "job1",
		Namespace: "cpod",
		JobType:   state.JobTypeMPI,
		JobStatus: state.JobStatusCreated,
		Extension: nil,
	}
	// job is timeout
	stateData := getStateData(time.Now().Add(-1*time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, true, false, false)
	s, e := parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sSucceed {
		t.Error("parse error")
	}
	stateData = getStateData(time.Now().Add(time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, true, false, false)
	s, e = parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sRunning {
		t.Error("parse error")
	}
	stateData = getStateData(time.Now().Add(time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, false, false, false)
	s, e = parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sCreated {
		t.Error("parse error")
	}
	stateData = getStateData(time.Now().Add(-1*time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, false, false, false)
	s, e = parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sFailed {
		t.Error("parse error")
	}

	stateData = getStateData(time.Now().Add(time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, true, true, false)
	s, e = parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sSucceed {
		t.Error("parse error")
	}
	stateData = getStateData(time.Now().Add(time.Minute).Format(config.TIME_FORMAT_FOR_K8S_LABEL), true, true, false, true)
	s, e = parseState(stateData)
	if e != nil {
		t.Error(e)
	}
	if s != sFailed {
		t.Error("parse error")
	}
}
