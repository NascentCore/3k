package errors

import "errors"

var (
	ErrCrdNotExist      = errors.New("crd not exists")
	ErrJobDownloading   = errors.New("job downloading")
	ErrJobComplete      = errors.New("job has completed")
	ErrCrdDataType      = errors.New("crd content data type error")
	ErrUnsupportedPhase = errors.New("unsupported phase")
)
