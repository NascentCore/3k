package fs

import (
	"fmt"
	"os"
	"path/filepath"
)

func IsDirExist(dir string) bool {
	_, err := os.Stat(dir)
	return !os.IsNotExist(err)
}

func RemoveAllFilesInDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		err = os.RemoveAll(filepath.Join(dir, name))
		if err != nil {
			return err
		}
	}
	return nil
}

// GetDirSize calculates the total size of files in a directory
func GetDirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}

// FormatBytes converts bytes to a human-readable format
func FormatBytes(bytes int64) string {
	const (
		_          = iota
		KB float64 = 1 << (10 * iota)
		MB
		GB
	)

	var size float64
	var unit string

	switch {
	case bytes >= int64(GB):
		size = float64(bytes) / GB
		unit = "GB"
	case bytes >= int64(MB):
		size = float64(bytes) / MB
		unit = "MB"
	default:
		size = float64(bytes) / KB
		unit = "KB"
	}

	return fmt.Sprintf("%.2f %s", size, unit)
}
