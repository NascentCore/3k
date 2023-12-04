package state

import "sxwl/3k/pkg/config"

// NO_TEST_NEEDED

type JobStatus string

// 通用的任务状态定义
// TODO： 结合更多的考虑，以及实际场景的经验，做细化调整
const (
	JobStatusCreated        JobStatus = "created"        //任务在Cpod中被创建（在K8S中被创建），pod在启动过程中
	JobStatusCreateFailed   JobStatus = "createfailed"   //任务在创建时直接失败（因为配置原因）
	JobStatusRunning        JobStatus = "running"        //Pod全部创建成功，并正常运行
	JobStatusPending        JobStatus = "pending"        //因为资源不足，在等待
	JobStatusErrorLoop      JobStatus = "crashloop"      //进入crashloop
	JobStatusModelUploaded  JobStatus = "modeluploaded"  //模型文件（训练结果）已上传
	JobStatusModelUploading JobStatus = "modeluploading" //模型文件（训练结果）正在上传
	JobStatusSucceed        JobStatus = "succeeded"      //所有工作成功完成
	JobStatusFailed         JobStatus = "failed"         //在中途以失败中止
	JobStatusUnknown        JobStatus = "unknown"        //无法获取任务状态，状态未知
)

type (
	JobType string
)

const (
	JobTypeMPI     JobType = config.PORTAL_JOBTYPE_MPI
	JobTypePytorch JobType = config.PORTAL_JOBTYPE_PYTORCH
	JobTypeGeneral JobType = config.PORTAL_JOBTYPE_GENERAL
)

type State struct {
	Name      string              `json:"name"`      //JobName
	Namespace string              `json:"namespace"` //K8S NameSpace
	JobType   JobType             `json:"jobtype"`
	JobStatus `json:"job_status"` //所处状态，
	Extension interface{}         `json:"extension"` //不同类型的任务有自己的扩展
}

// 判断一个状态是否不会再有任何变化
func (j JobStatus) NoMoreChange() bool {
	if j == JobStatusCreateFailed || j == JobStatusSucceed || j == JobStatusFailed {
		return true
	}
	return false
}
