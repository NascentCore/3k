package fs

import (
	"path/filepath"
	"strings"
)

func FileNameWithoutExtension(path string) string {
	fileNameWithExtension := filepath.Base(path)
	extension := filepath.Ext(fileNameWithExtension)
	fileNameWithoutExtension := strings.TrimSuffix(fileNameWithExtension, extension)

	return fileNameWithoutExtension
}
