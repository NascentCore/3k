package user

import (
	"sxwl/3k/pkg/uuid"
)

func NewUserID() (string, error) {
	return uuid.WithPrefix("user")
}
