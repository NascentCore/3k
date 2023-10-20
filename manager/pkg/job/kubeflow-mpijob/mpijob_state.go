package kubeflowmpijob

import (
	"fmt"
	"sxwl/3k/manager/pkg/job/state"
)

// NO_TEST_NEEDED

func GetState(namespace string) []state.State {
	data, err := listMPIJob(namespace)
	if err != nil {
		fmt.Println(err)
		return []state.State{}
	}
	//从Data中提取State信息
	res := []state.State{}
	for _, item := range data {
		item_, ok := item.(map[string]interface{})
		if !ok {
			fmt.Println("not map string interface")
			return []state.State{}
		}
		s := state.State{}
		status_, ok := item_["status"].(map[string]interface{})
		if !ok {
			continue
		}
		conditions_, ok := status_["conditions"].([]interface{})
		if !ok {
			continue
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
		} else {
			s.JobStatus = state.JobStatusCreateFailed
		}
		s.Namespace = namespace
		metadata_, ok := item_["metadata"].(map[string]interface{})
		if !ok {
			continue
		}
		name_, ok := metadata_["name"].(string)
		if !ok {
			continue
		}
		s.Name = name_

		res = append(res, s)
	}
	return res
}
