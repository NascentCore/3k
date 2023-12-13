package nascentcore.ai.modules.system.task;

import nascentcore.ai.config.SystemConfig;
import nascentcore.ai.modules.system.domain.Config;
import nascentcore.ai.modules.system.service.ConfigService;
import nascentcore.ai.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class AiadminTaskStartupRunner implements ApplicationRunner {
    @Autowired
    private ConfigService configService;
    @Override
    public void run(ApplicationArguments args){
        // 1. 读取数据库全部配置信息
        Map<String, String> configs = configService.queryAllValue();
        if (configs.containsKey("password")) {
            SystemConfig.setConfigs(configs);
        } else {
            //生产随机密钥
            String passwr = StringUtils.generateRandomString(8);
            Config config = new Config();
            config.setKeyName("password");
            config.setKeyValue(passwr);
            configs.put(config.getKeyName(),config.getKeyValue());
            configService.create(config);
            SystemConfig.setConfigs(configs);
        }
    }
}
