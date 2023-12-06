package nascentcore.ai.modules.system.service;

import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.utils.PageResult;

import java.util.List;
import java.util.Set;

/**
 * @description 服务接口
 * @author jim
 * @date 2023-10-12
 **/
public interface UserJobService extends IService<UserJob> {

    /**
     * 查询数据分页
     * @param criteria 条件
     * @param page 分页参数
     * @return PageResult
     */
    PageResult<UserJob> queryAll(UserJobQueryCriteria criteria, Page<Object> page);

    /**
     * 创建
     * @param resources /
     */
    void create(UserJob resources);

    /**
     * 查询所有数据
     * @param criteria 条件参数
     * @return List<UserJobDto>
     */
    List<UserJob> queryAll(UserJobQueryCriteria criteria);

    /**
     * 批量保存
     * @param columnInfos /
     */
    void save(List<UserJob> columnInfos);

    /**
     * 删除任务
     * @param ids /
     */
    void delete(Set<Long> ids);

    /**
     * 通过jobname删除任务
     * @param name /
     */
    void deletebyName(String name);
}
