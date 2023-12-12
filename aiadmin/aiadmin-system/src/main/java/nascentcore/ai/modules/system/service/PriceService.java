package nascentcore.ai.modules.system.service;

import java.util.List;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.Price;
import nascentcore.ai.modules.system.domain.vo.PriceQueryCriteria;
import nascentcore.ai.utils.PageResult;

public interface PriceService extends IService<Price> {

    /**
     * 查询数据分页
     * @param criteria 条件
     * @param page 分页参数
     * @return PageResult
     */
    PageResult<Price> queryAll(PriceQueryCriteria criteria, Page<Object> page);

    /**
     * 查询所有数据不分页
     * @param criteria 条件参数
     * @return List<PriceDto>
     */
    List<Price> queryAll(PriceQueryCriteria criteria);

    /**
     * 查询单个数据
     * @param gpuprod
     * @return PriceDto
     */
    Price queryByprod(String gpuprod);

    /**
     * 创建
     * @param resources /
     */
    void create(Price resources);

    /**
     * 编辑
     * @param resources /
     */
    void update(Price resources);

}
