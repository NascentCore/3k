package time

import (
	"time"
)

func GetNearestMinute(t time.Time) time.Time {
	nearestMinute := time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), 0, 0, t.Location())
	return nearestMinute
}
