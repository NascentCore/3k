package generaljob

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/log"
	"time"

	apiv1 "k8s.io/api/batch/v1"
)

func GetStates(namespace string) ([]state.State, error) {
	data, err := clientgo.GetK8SJobs(namespace)
	if err != nil {
		return []state.State{}, err
	}
	//从Data中提取State信息
	res := []state.State{}
	for _, item := range data.Items {
		// filter with label jobtype = generaljob
		if item.Labels["jobtype"] != "generaljob" {
			continue
		}
		s, err := parseState(&item)
		if err != nil {
			return []state.State{}, err
		}
		res = append(res, s)
	}
	return res, nil
}

func GetState(namespace, name string) (state.State, error) {
	data, err := clientgo.GetK8SJob(namespace, name)
	if err != nil {
		return state.State{}, err
	}
	s, err := parseState(data)
	if err != nil {
		log.SLogger.Errorw("parse generaljob state err", "error", err, "data", data)
		return state.State{}, err
	}
	return s, err
}

func parseState(data *apiv1.Job) (state.State, error) {
	s := state.State{JobType: state.JobTypeGeneral}
	s.Name = data.Name
	s.Namespace = data.Namespace
	//判断是否已经到截止时间
	timeup := false
	labels := data.Labels
	dl, ok := labels["deadline"]
	if ok {
		t, err := time.Parse(config.TIME_FORMAT_FOR_K8S_LABEL, dl)
		if err != nil {
			log.SLogger.Warnw("deadline parse err", "str", dl, "error", err)
		} else {
			if t.Before(time.Now()) { // job time is up
				timeup = true
			}
		}
	} else {
		log.SLogger.Warnw("labels have no field of deadline")
	}
	conditions := data.Status.Conditions
	condMap := map[string]string{}
	for _, cond := range conditions {
		condMap[string(cond.Type)] = string(cond.Status)
	}
	// TODO: add running status
	if condMap["Failed"] == "True" {
		s.JobStatus = state.JobStatusFailed
	} else if condMap["Complete"] == "True" {
		s.JobStatus = state.JobStatusSucceed
	} else { // created
		s.JobStatus = state.JobStatusCreated
	}

	if !s.JobStatus.NoMoreChange() { //还在运行中
		if timeup {
			//if s.JobStatus == state.JobStatusRunning { // running to succeed
			s.JobStatus = state.JobStatusSucceed
			//} else { // others to failed
			//s.JobStatus = state.JobStatusFailed
			//}
		}
	}
	return s, nil
}
