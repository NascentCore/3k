package sxwlzero

import (
	"fmt"
)

const (
	ErrDefaultCode uint32 = 400
)

type Error struct {
	ErrCode uint32 `json:"errCode"`
	ErrMsg  string `json:"errMsg"`
}

func NewError(errCode uint32, errMsg string) *Error {
	return &Error{ErrCode: errCode, ErrMsg: errMsg}
}

func (e *Error) GetCode() uint32 {
	return e.ErrCode
}

func (e *Error) GetMsg() string {
	return e.ErrMsg
}

func (e *Error) Error() string {
	return fmt.Sprintf("ErrCode=%d ErrMsg=%s", e.ErrCode, e.ErrMsg)
}

// Is compare the two Errors have same errCode
func (e *Error) Is(ee *Error) bool {
	return e.ErrCode == ee.ErrCode
}
