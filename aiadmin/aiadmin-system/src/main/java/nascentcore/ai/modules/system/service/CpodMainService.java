package nascentcore.ai.modules.system.service;

import nascentcore.ai.modules.system.domain.CpodMain;
import nascentcore.ai.modules.system.domain.User;
import nascentcore.ai.modules.system.domain.vo.CpodMainQueryCriteria;
import java.util.Map;
import java.util.List;
import java.io.IOException;
import javax.servlet.http.HttpServletResponse;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.utils.PageResult;

/**
* @description 服务接口
* @author jimi
* @date 2023-10-23
**/
public interface CpodMainService extends IService<CpodMain> {

    /**
    * 查询所有数据不分页
    * @param criteria 条件参数
    * @return List<CpodMainDto>
    */
    List<CpodMain> queryAll(CpodMainQueryCriteria criteria);

    /**
    * 创建
    * @param resources /
    */
    void create(CpodMain resources);

    /**
    * 编辑
    * @param resources /
    */
    void update(CpodMain resources);

    /**
     * 根据Cpodid查询
     *
     * @param cpodid /
     * @return /
     */
    List<CpodMain> findByCpodId(String cpodid);

    /**
     * 查询GPU类型
     *
     * @return /
     */
    List<CpodMain> queryAllGpuType();
}
