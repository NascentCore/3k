package job

// NO_TEST_NEEDED

import (
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	"sxwl/3k/manager/pkg/job/state"
	"sxwl/3k/manager/pkg/log"
)

func GetJobStates() ([]state.State, error) {
	res := []state.State{}
	//对于不同类型的任务，分别获取其任务状态，加入到结果中
	// KubeflowMPI
	if data, err := kubeflowmpijob.GetStates(JobNamespace); err == nil {
		res = append(res, data...)
	} else {
		log.SLogger.Errorw("err when get mpijob states", "error", err)
		return []state.State{}, err
	}
	// more if exists
	return res, nil
}
