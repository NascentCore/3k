package nascentcore.ai.modules.system.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import nascentcore.ai.modules.system.domain.Role;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

/**
 * @author jim
 * @date 2023-11-01
 */
@Mapper
public interface RoleMapper extends BaseMapper<Role> {

    List<Role> findByUserId(@Param("userId") Long userId);

}
