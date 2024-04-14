package email

import (
	"fmt"
	"io"
	"strings"

	"gopkg.in/mail.v2"
)

type SMTPClient struct {
	config    *Config
	templates map[string]string // Holds named templates
}

// NewSMTPClient creates a new instance of an SMTPClient with the given configuration.
func NewSMTPClient(cfg *Config) *SMTPClient {
	return &SMTPClient{
		config:    cfg,
		templates: make(map[string]string),
	}
}

// AddTemplate adds a new template with a given name.
func (client *SMTPClient) AddTemplate(name string, tmpl io.Reader) error {
	templateData, err := io.ReadAll(tmpl)
	if err != nil {
		return err
	}
	client.templates[name] = string(templateData)
	return nil
}

// SendPlainText sends a plain text email.
func (client *SMTPClient) SendPlainText(to []string, subject, body string) error {
	return sendEmail(client.config, to, subject, body, "text/plain")
}

// SendTemplateEmail sends an email using a named template.
func (client *SMTPClient) SendTemplateEmail(to []string, subject, templateName string, data interface{}) error {
	templateData, ok := client.templates[templateName]
	if !ok {
		return fmt.Errorf("template %s not found", templateName)
	}

	emailBody, err := ParseTemplate(strings.NewReader(templateData), data)
	if err != nil {
		return err
	}

	return sendEmail(client.config, to, subject, emailBody, "text/html")
}

// sendEmail handles the actual sending of the email.
func sendEmail(cfg *Config, to []string, subject, body, contentType string) error {
	m := mail.NewMessage()
	fromHeader := cfg.SenderName + " <" + cfg.Username + ">"
	m.SetHeader("From", fromHeader)
	m.SetHeader("To", to...)
	m.SetHeader("Subject", subject)
	m.SetBody(contentType, body)

	d := mail.NewDialer(cfg.Host, cfg.Port, cfg.Username, cfg.Password)
	return d.DialAndSend(m)
}
