package logic

import (
	"errors"
)

// 系统错误
var (
	ErrNotAdmin = errors.New("您没有管理员权限")
	ErrDBFind   = errors.New("数据库查询错误")
	ErrDB       = errors.New("数据库错误")
	ErrSystem   = errors.New("系统错误，请稍后再试")
)

var (
	MsgOssSyncBegin = "oss同步开始"
)

// 登录注册
var (
	ErrPassword    = errors.New("用户名或密码不正确")
	ErrCode        = errors.New("输入的验证码有误，请检查后重试。")
	ErrCodeTimeout = errors.New("验证码已过期，请重新获取。")
)

// 支付
var (
	ErrBalanceAddFail  = errors.New("余额充值失败")
	ErrBalanceFindFail = errors.New("余额查询失败")
)

const (
	MsgBalanceAddSuccess = "余额充值成功"
	MsgJobsDelSuccess    = "全部任务已终止"
)

const (
	DescRechargeRegister = "注册充值"
	DescRechargeAdmin    = "管理员充值"
)

// jupyterlab
var (
	ErrJupyterNotFound  = errors.New("jupyterlab不存在")
	ErrJupyterNotOwner  = errors.New("jupyterlab不属于您")
	ErrJupyterNoUpdate  = errors.New("jupyterlab没有更新的属性")
	ErrJupyterNotPaused = errors.New("jupyterlab不是已暂停状态，不能更新")
)

const (
	MsgJupyterStop   = "jupyterlab已暂停"
	MsgJupyterResume = "jupyterlab已启动"
	MsgJupyterUpdate = "jupyterlab已更新"
)

// resource
var (
	MsgResourceDeleteOK       = "资源删除成功"
	MsgResourceAddOK          = "资源创建成功"
	ErrResourceSyncTaskExists = errors.New("同步任务已存在")
	ErrModelNotFound          = errors.New("模型不存在")
	ErrDatasetNotFound        = errors.New("数据集不存在")
)

// app
var (
	ErrAppDuplicate    = errors.New("已经有该类型的应用实例")
	ErrAppNotExists    = errors.New("应用不存在")
	ErrAppHasJobs      = errors.New("有运行的实例，无法删除")
	MsgAppAddOK        = "应用实例创建成功"
	MsgAppUnregisterOK = "应用注销成功"
)

// cluster
var (
	ErrCpodNotFound = errors.New("集群不存在")
)

// job
var (
	ErrPermissionDenied = errors.New("无权访问该任务")
)
