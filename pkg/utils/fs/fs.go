package fs

import (
	"fmt"
	"math"
	"os"
)

func Exists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func BytesToMi(bytes int64) string {
	// Calculate the number of mebibytes as a float to use math.Ceil
	mib := float64(bytes) / (1024 * 1024)
	// Use math.Ceil to round up to the nearest whole number and convert back to int
	return fmt.Sprintf("%dMi", int(math.Ceil(mib)))
}
