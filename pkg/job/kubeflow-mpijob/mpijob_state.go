package kubeflowmpijob

import (
	"errors"
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/log"
	"time"
)

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
		log.SLogger.Errorw("get mpijob state err", "error", err)
		return state.State{}, err
	}
	s, err := parseState(data)
	if err != nil {
		log.SLogger.Errorw("parse mpijob state err", "error", err, "data", data)
		return state.State{}, err
	}
	return s, err
}

func parseState(data map[string]interface{}) (state.State, error) {
	s := state.State{JobType: state.JobTypeMPI}
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
	//判断是否已经到截止时间
	timeup := false
	labels, ok := metadata_["labels"].(map[string]interface{})
	if ok {
		dl, ok := labels["deadline"].(string)
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
	} else {
		log.SLogger.Warnw("metadata have no labels")
	}

	//从Data中提取State信息
	//When status is not there , maybe something wrong in job creation
	//mpijob is created , but cant run
	status, ok := data["status"].(map[string]interface{})
	if !ok {
		log.SLogger.Warnw("no status in kubeflow mpijob data")
		s.JobStatus = state.JobStatusUnknown
	} else {
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
		// ConditionType defination can be found in :
		// https://github.com/kubeflow/mpi-operator/blob/4a63d3cb35454d072c63fc84aeb5766878701ead/pkg/apis/kubeflow/v2beta1/types.go#L286
		// 在上面还定义了两种状态 Suspended 和 Restarting ， Restarting在MPIOperater的代码中没有用到，Suspended在现在的任务中不会出现
		if condMap["Failed"] == "True" { // check failed
			s.JobStatus = state.JobStatusFailed
		} else if condMap["Succeeded"] == "True" { // check succeed
			s.JobStatus = state.JobStatusSucceed
		} else if condMap["Running"] == "True" { // check running
			s.JobStatus = state.JobStatusRunning
		} else if condMap["Created"] == "True" { //  check created
			s.JobStatus = state.JobStatusCreated
		} else { //beside all above , then create failed
			s.JobStatus = state.JobStatusCreateFailed
		}
	}
	if !s.JobStatus.NoMoreChange() { //还在运行中
		if timeup {
			if s.JobStatus == state.JobStatusRunning { // running to succeed
				s.JobStatus = state.JobStatusSucceed
			} else { // others to failed
				s.JobStatus = state.JobStatusFailed
			}
		}
	}
	return s, nil
}
