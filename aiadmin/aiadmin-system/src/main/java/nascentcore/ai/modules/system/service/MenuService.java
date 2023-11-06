package nascentcore.ai.modules.system.service;

import com.baomidou.mybatisplus.extension.service.IService;
import nascentcore.ai.modules.system.domain.Menu;
import nascentcore.ai.modules.system.domain.vo.MenuVo;
import java.util.List;

public interface MenuService extends IService<Menu> {
    /**
     * 构建菜单树
     * @param menus 原始数据
     * @return /
     */
    List<Menu> buildTree(List<Menu> menus);

    /**
     * 构建菜单树
     * @param menus /
     * @return /
     */
    List<MenuVo> buildMenus(List<Menu> menus);


    /**
     * 根据当前用户获取菜单
     * @param currentUserId /
     * @return /
     */
    List<Menu> findByUser(Long currentUserId);
}
