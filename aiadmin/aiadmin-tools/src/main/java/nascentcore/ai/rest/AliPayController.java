package nascentcore.ai.rest;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import nascentcore.ai.domain.vo.TradeVo;
import nascentcore.ai.domain.AlipayConfig;
import nascentcore.ai.utils.AlipayUtils;
import nascentcore.ai.service.AliPayService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;


@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/aliPay")
@Api(tags = "工具：支付宝管理")
public class AliPayController {

    private final AlipayUtils alipayUtils;
    private final AliPayService alipayService;


    @ApiOperation("当面付支付")
    @PostMapping(value = "/toPayAsPrecreate")
    public ResponseEntity<String> toPayAsPrecreate(@Validated @RequestBody TradeVo trade) throws Exception {
        AlipayConfig alipay = alipayService.find();
        trade.setOutTradeNo(alipayUtils.getOrderCode());
        String payUrl = alipayService.toPayAsPrecreate(alipay, trade);
        return ResponseEntity.ok(payUrl);
    }
}
