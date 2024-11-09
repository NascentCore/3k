package model

const (
	BillingStatusContinue = 0 // 未结清
	BillingStatusComplete = 1 // 已结清
	BillingStatusUnpaid   = 0 // 未支付
	BillingStatusPaid     = 1 // 已支付
)

const (
	StrStatusNotAssigned   = "notassigned"   // 未下发
	StrStatusAssigned      = "assigned"      // 已下发
	StrStatusDataPreparing = "datapreparing" // 数据准备中
	StrStatusPending       = "pending"       // 启动中
	StrStatusPaused        = "paused"        // 已暂停
	StrStatusPausing       = "pausing"       // 暂停中
	StrStatusRunning       = "running"       // 运行中
	StrStatusFailed        = "failed"        // 运行失败
	StrStatusSucceeded     = "succeeded"     // 运行成功
	StrStatusStopped       = "stopped"       // 已终止
)

const (
	StatusNotAssigned   = 0 // 未下发
	StatusAssigned      = 1 // 已下发
	StatusDataPreparing = 2 // 数据准备中
	StatusPending       = 3 // 启动中
	StatusPaused        = 4 // 已暂停
	StatusPausing       = 5 // 暂停中
	StatusRunning       = 6 // 运行中
	StatusFailed        = 7 // 运行失败
	StatusSucceeded     = 8 // 运行成功
	StatusStopped       = 9 // 已终止
)

func FinalStatus(status int64) bool {
	switch status {
	case StatusStopped, StatusFailed, StatusSucceeded:
		return true
	default:
		return false
	}
}

const (
	StatusObtainNeedSend    = 0 // 需要下发
	StatusObtainNotNeedSend = 1 // 不需要下发
)

const (
	ReplicasRunning = 1
	ReplicasStop    = 0
)

var StatusToInt = map[string]int{
	StrStatusNotAssigned:   StatusNotAssigned,
	StrStatusAssigned:      StatusAssigned,
	StrStatusDataPreparing: StatusDataPreparing,
	StrStatusPending:       StatusPending,
	StrStatusPaused:        StatusPaused,
	StrStatusPausing:       StatusPausing,
	StrStatusRunning:       StatusRunning,
	StrStatusFailed:        StatusFailed,
	StrStatusSucceeded:     StatusSucceeded,
	StrStatusStopped:       StatusStopped,
}

var StatusToStr = map[int64]string{
	StatusNotAssigned:   StrStatusNotAssigned,
	StatusAssigned:      StrStatusAssigned,
	StatusDataPreparing: StrStatusDataPreparing,
	StatusPending:       StrStatusPending,
	StatusPaused:        StrStatusPaused,
	StatusPausing:       StrStatusPausing,
	StatusRunning:       StrStatusRunning,
	StatusFailed:        StrStatusFailed,
	StatusSucceeded:     StrStatusSucceeded,
	StatusStopped:       StrStatusStopped,
}
