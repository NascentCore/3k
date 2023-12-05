package nascentcore.ai.utils;

import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;

public class BaseHttp {

    public static String doHttp(String url, String method, String json, Map<String,String> headerMap) {
        String resp;
        try {
            URL httpUrl = new URL(url);
            if(url.startsWith("https")){
                HttpsURLConnection http= (HttpsURLConnection) httpUrl.openConnection();
                /**开启Https**/
                TrustManager[] tm = { new MyX509TrustManager() };
                SSLContext sslContext = SSLContext.getInstance("SSL", "SunJSSE");
                sslContext.init(null, tm, new java.security.SecureRandom());
                SSLSocketFactory ssf = sslContext.getSocketFactory();
                http.setSSLSocketFactory(ssf);
                resp = doConnectionBase(http,method,json,headerMap);
            }else{
                HttpURLConnection http= (HttpURLConnection) httpUrl.openConnection();
                resp = doConnectionBase(http,method,json,headerMap);
            }
        } catch (Exception e) {
            resp = null;
        }
        return resp;
    }

    /**
     * 发起连接（基础）
     */
    private static String doConnectionBase(HttpURLConnection http, String method,String json, Map<String, String> headerMap) throws IOException {
        String resp;
        // 连接超时
        http.setConnectTimeout(25000);
        // 读取超时 --服务器响应比较慢，增大时间
        http.setReadTimeout(25000);
        if(null == headerMap||!headerMap.containsKey("Content-Type")){
            //默认使用application/x-www-form-urlencoded
            http.setRequestProperty("Content-Type","application/x-www-form-urlencoded");
        }
        for (String key : headerMap.keySet()) {
            http.setRequestProperty(key,headerMap.get(key));
        }
        http.setInstanceFollowRedirects(true);

        http.setDoOutput(true);
        http.setDoInput(true);
        http.setUseCaches(false);
        http.setRequestMethod(method);
        http.connect();
        if(!StringUtils.isEmpty(json)){
            try (OutputStream os = http.getOutputStream()) {
                os.write(json.getBytes("UTF-8"));
            }
        }
        int state=http.getResponseCode();
        if(http.HTTP_OK==state){//请求成功
            resp = getString(http.getInputStream());
        }else{
            resp = getString(http.getErrorStream());
        }
        if (http != null) {
            http.disconnect();
        }
        return resp;
    }


    public static String getString(InputStream is) throws IOException {
        StringBuffer result = new StringBuffer();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is,
                "UTF-8"));
        String line = "";
        while ((line = reader.readLine()) != null) {
            result.append(line);
        }
        return result.toString();
    }
}
