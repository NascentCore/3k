package logic

import (
    "errors"
)

var (
    ErrNotAdmin = errors.New("您没有管理员权限")
    ErrDBFind   = errors.New("数据库查询错误")
)

var (
    ErrBalanceAddFail  = errors.New("余额充值失败")
    ErrBalanceFindFail = errors.New("余额查询失败")
)

var (
    MsgBalanceAddSuccess = "余额充值成功"
    MsgJobsDelSuccess    = "全部任务已终止"
)
