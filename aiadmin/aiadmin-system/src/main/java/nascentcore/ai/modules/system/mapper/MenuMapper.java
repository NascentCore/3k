package nascentcore.ai.modules.system.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import nascentcore.ai.modules.system.domain.Menu;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * @author jim
 * @date 2023-11-01
 */
@Mapper
public interface MenuMapper extends BaseMapper<Menu> {
    LinkedHashSet<Menu> findByRoleIdsAndTypeNot(@Param("roleIds") Set<Long> roleIds, @Param("type") Integer type);
}
