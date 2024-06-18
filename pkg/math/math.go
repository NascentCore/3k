package math

import (
	"math"
)

// Round rounds val to the nearest multiple of 1 / factor
func Round(val float64, places int) float64 {
	factor := math.Pow(10, float64(places))
	return math.Round(val*factor) / factor
}
