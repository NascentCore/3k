package errors

import (
	"fmt"
)

// UnImpl returns an error that indicate the function named as the input
// is not implemented yet.
func UnImpl(name string) error {
	return fmt.Errorf("%s is not implemented yet", name)
}
