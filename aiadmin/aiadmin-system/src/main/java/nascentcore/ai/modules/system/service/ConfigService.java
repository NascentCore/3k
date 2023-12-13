package nascentcore.ai.modules.system.service;

import java.util.List;
import java.util.Map;
import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.vo.ConfigQueryCriteria;
import nascentcore.ai.modules.system.domain.Config;

public interface ConfigService extends IService<Config> {
    /**
     * 查询所有数据
     * @param criteria 条件参数
     * @return List<Config>
     */
    List<Config> queryAll(ConfigQueryCriteria criteria);
    /**
     * 查询所有数据
     * @return Map<String, String>
     */
    Map<String, String> queryAllValue();

    /**
     * 创建
     * @param resources /
     */
    void create(Config resources);
}
