package nascentcore.ai.modules.system.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.modules.system.domain.Config;
import nascentcore.ai.modules.system.domain.vo.ConfigQueryCriteria;
import nascentcore.ai.modules.system.mapper.ConfigMapper;
import nascentcore.ai.modules.system.service.ConfigService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class ConfigServiceImpl extends ServiceImpl<ConfigMapper, Config> implements ConfigService {

    private final ConfigMapper configMapper;

    @Override
    public List<Config> queryAll(ConfigQueryCriteria criteria){
        return configMapper.findAll(criteria);
    }
    @Override
    public Map<String, String> queryAllValue() {
        ConfigQueryCriteria criteria = new ConfigQueryCriteria();
        List<Config> configList = configMapper.findAll(criteria);
        Map<String, String> systemConfigs = new HashMap<>();
        for (Config item : configList) {
            systemConfigs.put(item.getKeyName(), item.getKeyValue());
        }
        return systemConfigs;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void create(Config resources) {
        save(resources);
    }
}
