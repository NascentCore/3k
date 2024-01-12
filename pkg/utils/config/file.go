package config

import (
	"os"
	"strings"
)

// Reads the content of 'f' and return a key value map of the config entries.
func Read(f string) (map[string]string, error) {
	text, err := os.ReadFile(f)
	if err != nil {
		return nil, err
	}
	return parse(string(text)), nil
}

func parse(text string) map[string]string {
	res := make(map[string]string)
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		kv := strings.Split(line, ":")
		res[kv[0]] = strings.TrimSpace(kv[1])
	}
	return res
}
