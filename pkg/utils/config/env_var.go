package config

// Reading environment variables as config entries

import (
	"os"
	"strings"
)

// Deprecated: Moved to src/utils/sys/env_var.go:EnvVars.
func GetEnvVars() map[string]string {
	envVars := make(map[string]string)
	for _, e := range os.Environ() {
		pair := strings.SplitN(e, "=", 2)
		varName := pair[0]
		varValue := ""
		if len(pair) > 1 {
			varValue = pair[1]
		}
		envVars[varName] = varValue
	}
	return envVars
}
