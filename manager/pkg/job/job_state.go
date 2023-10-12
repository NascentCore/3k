package job

// NO_TEST_NEEDED

import (
	kubeflowmpijob "sxwl/3k/manager/pkg/job/kubeflow-mpijob"
	"sxwl/3k/manager/pkg/job/state"
)

func GetJobState() []state.State {
	res := []state.State{}
	//对于不同类型的任务，分别获取其任务状态，加入到结果中
	// KubeflowMPI
	res = append(res, kubeflowmpijob.GetState("cpod")...)
	// more if exists
	return res
}
