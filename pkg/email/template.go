package email

import (
	"bytes"
	"html/template"
	"io"
)

// ParseTemplate parses an HTML template from an io.Reader and injects data.
func ParseTemplate(tmpl io.Reader, data interface{}) (string, error) {
	templateData, err := io.ReadAll(tmpl)
	if err != nil {
		return "", err
	}
	t, err := template.New("email").Parse(string(templateData))
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}
