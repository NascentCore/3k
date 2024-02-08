package oss

import (
	"errors"
	"strings"
)

func ExtractURL(url string) (bucket, object string, err error) {
	// Check if the string starts with 'oss://'
	if !strings.HasPrefix(url, "oss://") {
		return "", "", errors.New("string does not start with 'oss://'")
	}

	// Remove the 'oss://' prefix
	trimmed := strings.TrimPrefix(url, "oss://")

	// Split the remaining string by '/'
	parts := strings.SplitN(trimmed, "/", 2)

	if len(parts) < 2 {
		return "", "", errors.New("string does not contain a valid 'bucket/object' format")
	}

	bucket = parts[0]
	object = parts[1]
	return
}
