package nascentcore.ai.modules.system.service.impl;

import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.modules.system.domain.Menu;
import nascentcore.ai.modules.system.domain.Role;
import nascentcore.ai.modules.system.domain.vo.MenuMetaVo;
import nascentcore.ai.modules.system.domain.vo.MenuVo;
import nascentcore.ai.modules.system.mapper.MenuMapper;
import nascentcore.ai.modules.system.service.MenuService;
import nascentcore.ai.modules.system.service.RoleService;
import nascentcore.ai.utils.*;
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author jim
 */
@Service
@RequiredArgsConstructor
@CacheConfig(cacheNames = "menu")
public class MenuServiceImpl extends ServiceImpl<MenuMapper, Menu> implements MenuService {

    private final MenuMapper menuMapper;
    private final RoleService roleService;

    @Override
    //@Cacheable(key = "'user:' + #p0")
    public List<Menu> findByUser(Long currentUserId) {
        List<Role> roles = roleService.findByUsersId(currentUserId);
        Set<Long> roleIds = roles.stream().map(Role::getId).collect(Collectors.toSet());
        LinkedHashSet<Menu> menus = menuMapper.findByRoleIdsAndTypeNot(roleIds, 2);
        return new ArrayList<>(menus);
    }

    @Override
    public List<Menu> buildTree(List<Menu> menus) {
        List<Menu> trees = new ArrayList<>();
        Set<Long> ids = new HashSet<>();
        for (Menu menu : menus) {
            if (menu.getPid() == null) {
                trees.add(menu);
            }
            for (Menu it : menus) {
                if (menu.getId().equals(it.getPid())) {
                    if (menu.getChildren() == null) {
                        menu.setChildren(new ArrayList<>());
                    }
                    menu.getChildren().add(it);
                    ids.add(it.getId());
                }
            }
        }
        if(trees.isEmpty()){
            trees = menus.stream().filter(s -> !ids.contains(s.getId())).collect(Collectors.toList());
        }
        return trees;
    }

    @Override
    public List<MenuVo> buildMenus(List<Menu> menus) {
        List<MenuVo> list = new LinkedList<>();
        menus.forEach(menu -> {
                    if (menu!=null){
                        List<Menu> menuList = menu.getChildren();
                        MenuVo menuVo = new MenuVo();
                        menuVo.setName(ObjectUtil.isNotEmpty(menu.getComponentName())  ? menu.getComponentName() : menu.getTitle());
                        // 一级目录需要加斜杠，不然会报警告
                        menuVo.setPath(menu.getPid() == null ? "/" + menu.getPath() :menu.getPath());
                        menuVo.setHidden(menu.getHidden());
                        // 如果不是外链
                        if(!menu.getIFrame()){
                            if(menu.getPid() == null){
                                menuVo.setComponent(StringUtils.isEmpty(menu.getComponent())?"Layout":menu.getComponent());
                                // 如果不是一级菜单，并且菜单类型为目录，则代表是多级菜单
                            }else if(menu.getType() == 0){
                                menuVo.setComponent(StringUtils.isEmpty(menu.getComponent())?"ParentView":menu.getComponent());
                            }else if(StringUtils.isNoneBlank(menu.getComponent())){
                                menuVo.setComponent(menu.getComponent());
                            }
                        }
                        menuVo.setMeta(new MenuMetaVo(menu.getTitle(),menu.getIcon(),!menu.getCache()));
                        if(CollectionUtil.isNotEmpty(menuList)){
                            menuVo.setAlwaysShow(true);
                            menuVo.setRedirect("noredirect");
                            menuVo.setChildren(buildMenus(menuList));
                            // 处理是一级菜单并且没有子菜单的情况
                        } else if(menu.getPid() == null){
                            MenuVo menuVo1 = new MenuVo();
                            menuVo1.setMeta(menuVo.getMeta());
                            // 非外链
                            if(!menu.getIFrame()){
                                menuVo1.setPath("index");
                                menuVo1.setName(menuVo.getName());
                                menuVo1.setComponent(menuVo.getComponent());
                            } else {
                                menuVo1.setPath(menu.getPath());
                            }
                            menuVo.setName(null);
                            menuVo.setMeta(null);
                            menuVo.setComponent("Layout");
                            if("".equals(menu.getPath()) || null == menu.getPath()){
                                menuVo.setRedirect("/"+menu.getPermission());
                            }
                            List<MenuVo> list1 = new ArrayList<>();
                            list1.add(menuVo1);
                            menuVo.setChildren(list1);
                        }
                        list.add(menuVo);
                    }
                }
        );
        return list;
    }
}
