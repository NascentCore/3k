
package nascentcore.ai.modules.system.service;

import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.Role;
import java.util.List;

/**
 * @author jim
 * @date 2023-11-01
 */
public interface RoleService extends IService<Role> {
    /**
     * 根据用户ID查询
     * @param userId 用户ID
     * @return /
     */
    List<Role> findByUsersId(Long userId);

}
