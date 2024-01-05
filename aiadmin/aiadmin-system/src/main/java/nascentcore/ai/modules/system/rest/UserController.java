package nascentcore.ai.modules.system.rest;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.annotation.rest.AnonymousPostMapping;
import nascentcore.ai.config.RsaProperties;
import nascentcore.ai.modules.system.domain.User;
import nascentcore.ai.modules.system.domain.vo.UserQueryCriteria;
import nascentcore.ai.modules.system.service.VerifyService;
import nascentcore.ai.utils.*;
import nascentcore.ai.modules.system.service.UserService;
import nascentcore.ai.utils.enums.CodeEnum;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import static cn.hutool.core.util.IdUtil.randomUUID;

@Api(tags = "系统：用户管理")
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {
    private final PasswordEncoder passwordEncoder;
    private final UserService userService;
    private final VerifyService verificationCodeService;

    @ApiOperation("注册用户")
    @AnonymousPostMapping(value = "/registerUser/{code}")
    public ResponseEntity<Object> registerUser(@PathVariable String code,  @RequestBody User resources) throws Exception {
        verificationCodeService.validated(CodeEnum.EMAIL_RESET_EMAIL_CODE.getKey() + resources.getEmail(), code);
        // 密码解密
        String password = RsaUtils.decryptByPrivateKey(RsaProperties.privateKey, resources.getPassword());
        resources.setPassword(passwordEncoder.encode(password));
        resources.setCompanyId(randomUUID());
        userService.create(resources);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @ApiOperation("查询用户")
    @GetMapping
    public ResponseEntity<PageResult<User>> queryUser(UserQueryCriteria criteria, Page<Object> page){
        Long userId = SecurityUtils.getCurrentUserId();
        UserQueryCriteria cri = new UserQueryCriteria();
        cri.setUserId(userId);
        return new ResponseEntity<>(userService.queryAll(cri,page),HttpStatus.OK);
    }
}
