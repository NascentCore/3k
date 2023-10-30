package nascentcore.ai.modules.system.rest;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.annotation.rest.AnonymousPostMapping;
import nascentcore.ai.domain.vo.EmailVo;
import nascentcore.ai.service.EmailService;
import nascentcore.ai.modules.system.service.VerifyService;
import nascentcore.ai.utils.enums.CodeEnum;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * @author jim
 * @date 2023-9-27
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/code")
@Api(tags = "系统：验证码管理")
public class VerifyController {
    private final VerifyService verificationCodeService;
    private final EmailService emailService;

    @AnonymousPostMapping(value = "/sendEmail")
    @ApiOperation("邮箱，发送验证码")
    public ResponseEntity<Object> sendEmail(@RequestParam String email){
        EmailVo emailVo = verificationCodeService.sendEmail(email, CodeEnum.EMAIL_RESET_EMAIL_CODE.getKey());
        emailService.send(emailVo,emailService.find());
        return new ResponseEntity<>(HttpStatus.OK);
    }
}
