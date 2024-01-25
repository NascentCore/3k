package job

import "github.com/google/uuid"

func NewJobName() (string, error) {
	newUUID, err := uuid.NewRandom()
	if err != nil {
		return "", err
	}

	return "ai" + newUUID.String(), nil
}
