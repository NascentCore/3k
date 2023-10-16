package utility

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"sxwl/3k/pkg/utils/consts"
)

func GetJobInfo(url string) ([]byte, error) {
	req, err := http.NewRequest(consts.GET, url, nil)
	if err != nil {
		return nil, fmt.Errorf("Failed to create HTTP request, error: %v", err)
	}
	req.Header.Add(consts.Authorization, "xxxx")
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Failed to do HTTP request: %v, error: %v", req, err)
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Failed to read all HTTP response, error: %v", err)
	}
	return body, nil
}
