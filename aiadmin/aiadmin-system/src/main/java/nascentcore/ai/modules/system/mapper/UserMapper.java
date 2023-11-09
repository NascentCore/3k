package nascentcore.ai.modules.system.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import nascentcore.ai.modules.system.domain.User;
import nascentcore.ai.modules.system.domain.vo.UserQueryCriteria;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import java.util.Date;
import java.util.List;
import java.util.Set;

/**
 * @author jim
 * @date 2023-09-27
 */
@Mapper
public interface UserMapper extends BaseMapper<User> {
    List<User> findAll(@Param("criteria") UserQueryCriteria criteria);
    Long countAll(@Param("criteria") UserQueryCriteria criteria);
    User findByUsername(@Param("username") String username);

    User findByEmail(@Param("email") String email);

    User findByPhone(@Param("phone") String phone);

}
