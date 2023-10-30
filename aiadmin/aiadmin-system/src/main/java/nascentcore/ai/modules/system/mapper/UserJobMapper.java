package nascentcore.ai.modules.system.mapper;

import nascentcore.ai.modules.system.domain.UserJob;
import nascentcore.ai.modules.system.domain.vo.UserJobQueryCriteria;
import java.util.List;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Mapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

/**
 * @author jim
 * @date 2023-10-12
 **/
@Mapper
public interface UserJobMapper extends BaseMapper<UserJob> {

    IPage<UserJob> findAll(@Param("criteria") UserJobQueryCriteria criteria, Page<Object> page);

    List<UserJob> findAll(@Param("criteria") UserJobQueryCriteria criteria);
}
