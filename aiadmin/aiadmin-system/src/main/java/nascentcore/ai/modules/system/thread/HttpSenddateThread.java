package nascentcore.ai.modules.system.thread;

import com.alibaba.fastjson.JSON;
import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.bean.Constants;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import nascentcore.ai.modules.system.service.FileurlService;
import nascentcore.ai.utils.BaseHttp;
import nascentcore.ai.utils.SpringContextHolder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HttpSenddateThread implements Runnable {
    private static boolean isStart = false;
    private final static List<UserJob> sendList =  new ArrayList<>();
    private final FileurlService fileurlService;

    public HttpSenddateThread() {
        fileurlService = SpringContextHolder.getBean(FileurlService.class);
    }

    public static void add(UserJob data) {
        synchronized (data) {
            sendList.add(data);
        }
        if (!isStart) {
            Thread thread = new Thread(new HttpSenddateThread());
            thread.start();
            isStart = true;
        }
    }

    @Override
    @SuppressWarnings("InfiniteLoopStatement")
    public void run() {
        while (true) {
            try {
                List<UserJob> list = sendList;
                if (!list.isEmpty()) {
                    List<UserJob> newList = new ArrayList<>();
                    synchronized (list) {
                        newList.addAll(list);
                        list.clear();
                    }
                    sendThirddata(newList);
                } else {
                    try {
                        Thread.sleep(10 * 1000);
                    } catch (InterruptedException e) {
                        isStart = false;
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void sendThirddata(List<UserJob> newList) {
        for (UserJob userJob : newList) {
            if (Constants.WORKER_STATUS_FIAL == userJob.getWorkStatus()) {
                Map<String, Object> obj = new HashMap<String, Object>();
                obj.put("status","fail");
                postApi(userJob.getCallbackUrl(), JSON.toJSONString(obj));
            } else if (Constants.WORKER_STATUS_URL_SUCCESS == userJob.getWorkStatus()) {
                FileurlQueryCriteria fileurlQueryCriteria = new FileurlQueryCriteria();
                fileurlQueryCriteria.setJobName(userJob.getJobName());
                List<Fileurl> fileurlList = fileurlService.queryAll(fileurlQueryCriteria);
                Map<String, Object> obj = new HashMap<String, Object>();
                obj.put("status","success");
                obj.put("url",fileurlList.get(0).getFileUrl());
                obj.put("jobId",fileurlList.get(0).getJobName());
                postApi(userJob.getCallbackUrl(), JSON.toJSONString(obj));
            }
        }
    }
    private static void postApi(String url, String json) {
        Map<String, String> headerMap = new HashMap<>();
        headerMap.put("Content-Type", "application/json");
        BaseHttp.doHttp(url, "POST", json, headerMap);
    }


}
