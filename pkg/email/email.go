package email

import "io"

// Emailer defines the interface for sending emails.
type Emailer interface {
	AddTemplate(name string, tmpl io.Reader) error
	SendPlainText(to []string, subject, body string) error
	SendTemplateEmail(to []string, subject, templateName string, data interface{}) error
}

// Config holds configuration for any SMTP client
type Config struct {
	Host       string
	Port       int
	Username   string
	SenderName string
	Password   string
}
