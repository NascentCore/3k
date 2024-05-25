package uuid

import (
    "fmt"
    
    "github.com/google/uuid"
)

func WithPrefix(prefix string) (string, error) {
    newUUID, err := uuid.NewRandom()
    if err != nil {
        return "", err
    }
    
    return fmt.Sprintf("%s-%s", prefix, newUUID.String()), nil
}
