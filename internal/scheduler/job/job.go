package job

import (
	"sxwl/3k/pkg/uuid"
)

func NewJobName() (string, error) {
	return uuid.UUIDWithPrefix("ai")
}
