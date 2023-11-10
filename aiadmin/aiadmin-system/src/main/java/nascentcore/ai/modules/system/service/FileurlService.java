package nascentcore.ai.modules.system.service;

import nascentcore.ai.modules.system.domain.Fileurl;
import nascentcore.ai.modules.system.domain.vo.FileurlQueryCriteria;
import java.util.List;
import com.baomidou.mybatisplus.extension.service.IService;

public interface FileurlService extends IService<Fileurl> {

    /**
     * 查询所有数据不分页
     * @param criteria 条件参数
     * @return List<FileurlDto>
     */
    List<Fileurl> queryAll(FileurlQueryCriteria criteria);

    /**
     * 批量保存
     * @param columnInfos /
     */
    void save(List<Fileurl> columnInfos);
}
