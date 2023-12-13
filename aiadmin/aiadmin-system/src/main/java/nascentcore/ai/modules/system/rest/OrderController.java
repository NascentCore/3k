package nascentcore.ai.modules.system.rest;

import cn.hutool.core.bean.BeanUtil;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import nascentcore.ai.annotation.rest.AnonymousPostMapping;
import nascentcore.ai.domain.AlipayConfig;
import nascentcore.ai.exception.BadRequestException;
import nascentcore.ai.modules.system.domain.Order;
import nascentcore.ai.modules.system.domain.vo.OrderQueryCriteria;
import nascentcore.ai.modules.system.domain.AliPayRes;
import nascentcore.ai.service.AliPayService;
import nascentcore.ai.modules.system.service.OrderService;
import nascentcore.ai.utils.*;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import org.slf4j.LoggerFactory;
@Api(tags = "支付：支付管理")
@RestController
@RequestMapping("/api/order")
@RequiredArgsConstructor
public class OrderController {
    private static final org.slf4j.Logger log = LoggerFactory.getLogger(OrderController.class);
    private final OrderService orderService;
    private final AlipayUtils alipayUtils;
    private final AliPayService alipayService;

    @GetMapping(value = "/order_status")
    @ApiOperation("获取此任务的支付状态")
    public ResponseEntity<AliPayRes> getJobPayStatus(String jobName) {
        OrderQueryCriteria orderQueryCriteria = new OrderQueryCriteria();
        orderQueryCriteria.setJobName(jobName);
        Order order = orderService.queryOne(orderQueryCriteria);
        AliPayRes aliPayRes = new AliPayRes();
        if (null != order) {
            if (order.getStatus() == AlipayUtils.STATUS_PAY) {
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_PAY));
            } else {
                aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_NOTPAY));
            }
        } else {
            aliPayRes.setState(String.valueOf(AlipayUtils.STATUS_NOTPAY));
        }
        return new ResponseEntity<>(aliPayRes, HttpStatus.OK);
    }

    @GetMapping(value = "/order_info")
    @ApiOperation("获取此任务的支付信息")
    public ResponseEntity<AliPayRes> getJobPayInfo(String jobName) {
        AliPayRes aliPayRes = orderService.getPayInfo(jobName);
        return new ResponseEntity<>(aliPayRes, HttpStatus.OK);
    }

    @GetMapping
    public ResponseEntity<PageResult<Order>> queryOrder(OrderQueryCriteria criteria, Page<Object> page) {
        Long userId = SecurityUtils.getCurrentUserId();
        OrderQueryCriteria cri = new OrderQueryCriteria();
        BeanUtil.copyProperties(criteria, cri);
        cri.setUserId(userId);
        return new ResponseEntity<>(orderService.queryAll(cri, page), HttpStatus.OK);
    }

    @AnonymousPostMapping("/alipay_notify")
    @ApiOperation("支付异步通知(要公网访问)，接收异步通知，检查通知内容app_id、out_trade_no、total_amount是否与请求中的一致，根据trade_status进行后续业务处理")
    public ResponseEntity<Object> notify(HttpServletRequest request) {
        AlipayConfig alipay = alipayService.find();
        try {
            // 对称解密
            alipay.setPrivateKey(EncryptUtils.desDecrypt(alipay.getPrivateKey()));
            alipay.setPublicKey(EncryptUtils.desDecrypt(alipay.getPublicKey()));
        } catch (Exception e) {
            throw new BadRequestException(e.getMessage());
        }
        //内容验签，防止黑客篡改参数
        if (alipayUtils.rsaCheck(request, alipay)) {
            //交易状态
            String tradeStatus = new String(request.getParameter("trade_status").getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8);
            // 商户订单号
            String outTradeNo = new String(request.getParameter("out_trade_no").getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8);
            //支付宝交易号
            String tradeNo = new String(request.getParameter("trade_no").getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8);
            //验证
            log.debug("交易状态：{}", tradeStatus);
            log.debug("outTradeNo：{}", outTradeNo);
            log.debug("tradeNo：{}", tradeNo);
            if (tradeStatus.equals(AliPayStatusEnum.SUCCESS.getValue()) || tradeStatus.equals(AliPayStatusEnum.FINISHED.getValue())) {
                // 验证通过后应该根据业务需要处理订单
                Order order = orderService.querybyNo(outTradeNo);
                if (null != order && AlipayUtils.STATUS_NOTPAY == order.getStatus()) {
                    order.setStatus(AlipayUtils.STATUS_PAY);
                    order.setTradeNo(tradeNo);
                    order.setUpdateTime(DateUtil.getUTCTimeStamp());
                    orderService.update(order);
                    return new ResponseEntity<>(HttpStatus.OK);
                }
            }
        }
        return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
    }
}
