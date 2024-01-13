package nascentcore.ai.modules.security.rest;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import nascentcore.ai.annotation.rest.AnonymousPostMapping;
import nascentcore.ai.config.RsaProperties;
import nascentcore.ai.domain.vo.EmailVo;
import nascentcore.ai.modules.security.config.bean.SecurityProperties;
import nascentcore.ai.modules.security.security.TokenProvider;
import nascentcore.ai.modules.security.service.dto.AuthUserDto;
import nascentcore.ai.modules.security.service.OnlineUserService;
import nascentcore.ai.modules.security.service.dto.JwtUserDto;
import nascentcore.ai.modules.system.service.VerifyService;
import nascentcore.ai.service.EmailService;
import nascentcore.ai.utils.RedisUtils;
import nascentcore.ai.utils.RsaUtils;
import nascentcore.ai.utils.SecurityUtils;
import nascentcore.ai.utils.enums.CodeEnum;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import javax.servlet.http.HttpServletRequest;
import java.util.HashMap;
import java.util.Map;

/**
 * @author jim
 * @date 2023-9-25
 * 授权、根据token获取用户详细信息
 */
@Slf4j
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@Api(tags = "系统：系统授权接口")
public class AuthorizationController {
    private final SecurityProperties properties;
    private final OnlineUserService onlineUserService;
    private final TokenProvider tokenProvider;
    private final AuthenticationManagerBuilder authenticationManagerBuilder;
    private final RedisUtils redisUtils;
    private final VerifyService verificationCodeService;
    private final EmailService emailService;

    @ApiOperation("登录授权")
    @AnonymousPostMapping(value = "/login")
    public ResponseEntity<Object> login(@Validated @RequestBody AuthUserDto authUser, HttpServletRequest request) throws Exception {
        // 密码解密
        String password = RsaUtils.decryptByPrivateKey(RsaProperties.privateKey, authUser.getPassword());

        UsernamePasswordAuthenticationToken authenticationToken =
                new UsernamePasswordAuthenticationToken(authUser.getUsername(), password);
        Authentication authentication = authenticationManagerBuilder.getObject().authenticate(authenticationToken);
        SecurityContextHolder.getContext().setAuthentication(authentication);
        if (redisUtils.hasKey(authUser.getUsername())) {
            String token = (String) redisUtils.get(authUser.getUsername());
            final JwtUserDto jwtUserDto = (JwtUserDto) authentication.getPrincipal();
            Map<String, Object> authInfo = new HashMap<String, Object>(2) {{
                put("token", properties.getTokenStartWith() + token);
                put("user", jwtUserDto);
            }};
            onlineUserService.save(jwtUserDto, token, request);
            return ResponseEntity.ok(authInfo);
        } else {
            // 生成 Token
            String token = tokenProvider.createToken(authentication);
            final JwtUserDto jwtUserDto = (JwtUserDto) authentication.getPrincipal();
            // 返回 token 与 用户信息
            Map<String, Object> authInfo = new HashMap<String, Object>(2) {{
                put("token", properties.getTokenStartWith() + token);
                put("user", jwtUserDto);
            }};
            // 保存 Token 到 Redis
            redisUtils.set(authUser.getUsername(), token);
            // 保存在线信息
            onlineUserService.save(jwtUserDto, token, request);
            EmailVo emailVo = verificationCodeService.sendTokenEmail(authUser.getUsername());
            emailService.send(emailVo, emailService.find());
            // 返回登录信息
            return ResponseEntity.ok(authInfo);
        }
    }

    @ApiOperation("退出登录")
    @DeleteMapping(value = "/logout")
    public ResponseEntity<Object> logout(HttpServletRequest request) {
        onlineUserService.logout(tokenProvider.getToken(request));
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @ApiOperation("获取用户信息")
    @GetMapping(value = "/info")
    public ResponseEntity<UserDetails> getUserInfo() {
        return ResponseEntity.ok(SecurityUtils.getCurrentUser());
    }

}
