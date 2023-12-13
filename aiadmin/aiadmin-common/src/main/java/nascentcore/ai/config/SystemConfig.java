package nascentcore.ai.config;

import java.util.HashMap;
import java.util.Map;

public class SystemConfig {
    //所有的配置均保存在该 HashMap 中
    private static Map<String, String> SYSTEM_CONFIGS = new HashMap<>();

    public static String getConfig(String keyName) {
        return SYSTEM_CONFIGS.get(keyName);
    }
    public static void setConfigs(Map<String, String> configs) {
        SYSTEM_CONFIGS = configs;
    }
}
