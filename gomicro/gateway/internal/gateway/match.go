package gateway

import (
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
)

var (
	GlobalMatcher *Matcher
)

type PassRoute struct {
	URL  string
	Auth bool
}

type Matcher struct {
	c Config
}

func NewMatcher(c Config) *Matcher {
	return &Matcher{c: c}
}

func (m *Matcher) Match(r *http.Request) (*PassRoute, error) {
	for _, route := range m.c.Routers {
		if strings.HasPrefix(path.Clean(r.URL.Path), route.Path) {
			server, ok := m.c.Servers[route.Server]
			if !ok {
				return nil, fmt.Errorf("can not find server:%s for path:%s", route.Server, r.URL.Path)
			}

			toPath := r.URL.Path
			if route.ToPath != "" {
				toPath = route.ToPath
			}
			toUrl, err := url.JoinPath(server, toPath)
			if err != nil {
				return nil, fmt.Errorf("can not find server:%s for path:%s", route.Server, r.URL.Path)
			}

			return &PassRoute{
				URL:  toUrl,
				Auth: route.Auth,
			}, nil
		}
	}

	return nil, fmt.Errorf("can not find router for path:%s", r.URL.Path)
}
