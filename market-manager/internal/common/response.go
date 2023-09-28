package common

import (
	"encoding/json"
)

var (
	// OK
	OK = NewError(0, "SUCCESS")

	// 服务级错误码
	ErrServer = NewError(10001, "server error, please try again")
	ErrParam  = NewError(10002, "parameter error, please check the parameter")
)

var _ Error = (*Err)(nil)

type Error interface {
	// i 为了避免被其他包实现
	i()
	// WithData 设置成功时返回的数据
	WithData(data interface{}) Error
	// WithID 设置当前请求的唯一ID
	WithMsg(id string) Error
	// ToString 返回 JSON 格式的错误详情
	ToString() string
}

type Err struct {
	Code int         `json:"code"` // 业务编码
	Msg  string      `json:"msg"`  // 错误描述
	Data interface{} `json:"data"` // 成功时返回的数据
}

func NewError(code int, msg string) Error {
	return &Err{
		Code: code,
		Msg:  msg,
		Data: nil,
	}
}

func (e *Err) i() {}

func (e *Err) WithData(data interface{}) Error {
	e.Data = data
	return e
}

func (e *Err) WithMsg(msg string) Error {
	e.Msg = msg
	return e
}

// ToString 返回 JSON 格式的错误详情
func (e *Err) ToString() string {
	err := &struct {
		Code int         `json:"code"`
		Msg  string      `json:"msg"`
		Data interface{} `json:"data"`
	}{
		Code: e.Code,
		Msg:  e.Msg,
		Data: e.Data,
	}

	raw, _ := json.Marshal(err)
	return string(raw)
}
