package kubeflowmpijob

import (
	"errors"
	"fmt"
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
	"sxwl/3k/manager/pkg/job/state"
)

// NO_TEST_NEEDED

var ErrNotFound error = errors.New("not found")

func GetStates(namespace string) ([]state.State, error) {
	data, err := listMPIJob(namespace)
	if err != nil {
		return []state.State{}, err
	}
	//从Data中提取State信息
	res := []state.State{}
	for _, item := range data {
		item_, ok := item.(map[string]interface{})
		if !ok {
			return []state.State{}, errors.New("data format err")
		}
		s, err := parseState(item_)
		if err != nil {
			return []state.State{}, err
		}
		res = append(res, s)
	}
	return res, nil
}

func GetState(namespace, name string) (state.State, error) {
	data, err := clientgo.GetObjectData(namespace, "kubeflow.org", "v2beta1", "mpijobs", name)
	if err != nil {
		//如果不存在返回 Not Found
		if err.Error() == fmt.Sprintf(`mpijobs.kubeflow.org "%s" not found`, name) {
			return state.State{}, ErrNotFound
		}
		return state.State{}, err
	}
	s, err := parseState(data)
	if err != nil {
		return state.State{}, err
	}
	return s, err
}

func parseState(data map[string]interface{}) (state.State, error) {
	s := state.State{}
	//从Data中提取State信息
	status, ok := data["status"].(map[string]interface{})
	if !ok {
		return state.State{}, errors.New("no status")
	}
	conditions_, ok := status["conditions"].([]interface{})
	if !ok {
		return state.State{}, errors.New("no conditions")
	}
	condMap := map[string]string{}
	for _, cond := range conditions_ {
		cond_, ok := cond.(map[string]interface{})
		if !ok {
			continue
		}
		ty, ok := cond_["type"].(string)
		if !ok {
			continue
		}
		st, ok := cond_["status"].(string)
		if !ok {
			continue
		}
		condMap[ty] = st
	}
	// TODO: analyze more data or find some doc , known more about mpijob status
	if condMap["Failed"] == "True" {
		s.JobStatus = state.JobStatusFailed
	} else if condMap["Running"] == "True" {
		s.JobStatus = state.JobStatusRunning
	} else if condMap["Created"] == "True" {
		s.JobStatus = state.JobStatusCreated
	} else if condMap["Succeeded"] == "True" {
		s.JobStatus = state.JobStatusSucceed
	} else {
		s.JobStatus = state.JobStatusCreateFailed
	}
	metadata_, ok := data["metadata"].(map[string]interface{})
	if !ok {
		return state.State{}, errors.New("no metadata")
	}
	name_, ok := metadata_["name"].(string)
	if !ok {
		return state.State{}, errors.New("invalid metadata")
	}
	s.Name = name_
	namespace_, ok := metadata_["namespace"].(string)
	if !ok {
		return state.State{}, errors.New("invalid metadata")
	}
	s.Namespace = namespace_
	return s, nil
}
