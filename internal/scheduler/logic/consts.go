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
