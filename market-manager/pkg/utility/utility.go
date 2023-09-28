package utility

import (
	"github.com/golang/glog"
	"io/ioutil"
	"net/http"
)

func GetJobInfo(url string) ([]byte, error) {
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Add("Authorization", "xxxx")
	response, err := http.DefaultClient.Do(req)
	defer response.Body.Close()
	if err != nil {
		glog.Error("get job info error: ", err)
		return nil, err
	}
	body, _ := ioutil.ReadAll(response.Body)
	return body, nil
}
