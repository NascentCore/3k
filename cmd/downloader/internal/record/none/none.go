package none

import (
	"sxwl/3k/cmd/downloader/internal/consts"
)

type Recorder struct {
}

func NewRecorder() *Recorder {
	return &Recorder{}
}

func (r *Recorder) Check() error {
	return nil
}

func (r *Recorder) Begin() error {
	return nil
}

func (r *Recorder) Fail() error {
	return nil
}

func (r *Recorder) Complete() error {
	return nil
}

func (r *Recorder) Name() string {
	return consts.None
}
