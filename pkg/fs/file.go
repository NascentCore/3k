package fs

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

func FileNameWithoutExtension(path string) string {
	fileNameWithExtension := filepath.Base(path)
	extension := filepath.Ext(fileNameWithExtension)
	fileNameWithoutExtension := strings.TrimSuffix(fileNameWithExtension, extension)

	return fileNameWithoutExtension
}

// TouchFile updates the modification time of the file at the specified path
// or creates an empty file if it does not exist.
func TouchFile(path string) error {
	// Get the current time
	currentTime := time.Now()

	// Try to open the file in read-write mode
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// Update the access and modification times of the file
	if err := os.Chtimes(path, currentTime, currentTime); err != nil {
		return err
	}

	return nil
}
