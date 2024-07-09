package storage

import (
	"fmt"
	"regexp"
	"strconv"
)

func HumanReadableToBytes(sizeStr string) (int64, error) {
	units := map[string]float64{
		"B":  1,
		"Ki": 1024,
		"Mi": 1024 * 1024,
		"Gi": 1024 * 1024 * 1024,
		"Ti": 1024 * 1024 * 1024 * 1024,
	}

	re := regexp.MustCompile(`^(\d+(?:\.\d+)?)([A-Za-z]*)$`)
	matches := re.FindStringSubmatch(sizeStr)
	if matches == nil || len(matches) < 3 {
		return 0, fmt.Errorf("invalid size format: %s", sizeStr)
	}

	numStr, unit := matches[1], matches[2]
	size, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, err
	}

	multiplier, found := units[unit]
	if !found {
		return 0, fmt.Errorf("unknown unit: %s", unit)
	}

	return int64(size * multiplier), nil
}

func BytesToHumanReadable(numBytes int64) string {
	units := []string{"B", "Ki", "Mi", "Gi", "Ti"}
	value := float64(numBytes)

	for _, unit := range units {
		if value < 1024 {
			return fmt.Sprintf("%.2f%s", value, unit)
		}
		value /= 1024
	}
	return fmt.Sprintf("%.2fTi", value)
}

func MBToBytes(mb int64) int64 {
	return mb * 1024 * 1024
}

func GBToBytes(gb int64) int64 {
	return gb * 1024 * 1024 * 1024
}
