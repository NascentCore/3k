-- MySQL dump 10.13  Distrib 8.0.35, for macos14.0 (arm64)
--
-- Host: 8.140.22.241    Database: aiadmin
-- ------------------------------------------------------
-- Server version	5.7.44

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `sys_cpod_cache`
--

DROP TABLE IF EXISTS `sys_cpod_cache`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_cpod_cache` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `cpod_id` varchar(255) NOT NULL DEFAULT '' COMMENT 'cpod id',
  `cpod_version` varchar(255) NOT NULL DEFAULT '' COMMENT 'pod 版本',
  `data_type` tinyint(4) NOT NULL DEFAULT '0' COMMENT '缓存的数据类型',
  `data_name` varchar(255) NOT NULL DEFAULT '' COMMENT '缓存的数据名字',
  `data_id` varchar(255) NOT NULL DEFAULT '' COMMENT '缓存的数据id',
  `data_size` bigint(20) NOT NULL DEFAULT '0' COMMENT '资源体积(字节)',
  `data_source` varchar(255) NOT NULL DEFAULT '' COMMENT '缓存的数据来源',
  `public` tinyint(4) NOT NULL DEFAULT '1' COMMENT '资源是否公开 1 公共 2 用户私有',
  `user_id` bigint(20) DEFAULT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) DEFAULT NULL COMMENT '用户ID',
  `template` varchar(255) NOT NULL DEFAULT '' COMMENT '模型推理模版',
  `finetune_gpu_count` tinyint(4) NOT NULL DEFAULT '1' COMMENT '微调需要最少GPU',
  `inference_gpu_count` tinyint(4) NOT NULL DEFAULT '1' COMMENT '推理需要最少GPU',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_cpod_id` (`cpod_id`),
  KEY `idx_data_id_type` (`data_id`,`data_type`,`cpod_id`)
) ENGINE=InnoDB AUTO_INCREMENT=13237 DEFAULT CHARSET=utf8mb4 COMMENT='cpod缓存资源表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_cpod_main`
--

DROP TABLE IF EXISTS `sys_cpod_main`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_cpod_main` (
  `main_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `cpod_id` varchar(60) DEFAULT NULL COMMENT 'cpod id',
  `cpod_version` varchar(60) DEFAULT NULL COMMENT 'pod 版本',
  `gpu_vendor` varchar(255) DEFAULT NULL COMMENT 'gpu vendor',
  `gpu_prod` varchar(255) DEFAULT NULL COMMENT 'GPU型号',
  `gpu_mem` bigint(20) DEFAULT NULL COMMENT 'GPU显存(MB)',
  `gpu_total` int(5) DEFAULT NULL COMMENT 'GPU总数量',
  `gpu_allocatable` int(5) DEFAULT NULL COMMENT 'GPU可分配数量',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `user_id` varchar(60) DEFAULT NULL COMMENT '算力源注册ID',
  PRIMARY KEY (`main_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=511 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='cpod主信息';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_cpod_node`
--

DROP TABLE IF EXISTS `sys_cpod_node`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_cpod_node` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'id',
  `cpod_id` varchar(60) NOT NULL DEFAULT '' COMMENT 'cpod id',
  `cpod_version` varchar(60) NOT NULL DEFAULT '' COMMENT 'pod 版本',
  `user_id` varchar(60) NOT NULL DEFAULT '' COMMENT '算力源注册ID',
  `node_name` varchar(60) NOT NULL DEFAULT '' COMMENT 'node名字',
  `gpu_vendor` varchar(255) NOT NULL DEFAULT '' COMMENT 'gpu vendor',
  `gpu_prod` varchar(255) NOT NULL DEFAULT '' COMMENT 'GPU型号',
  `gpu_mem` bigint(20) NOT NULL DEFAULT '0' COMMENT 'GPU显存(bytes)',
  `gpu_total` int(5) NOT NULL DEFAULT '0' COMMENT 'GPU总数量',
  `gpu_allocatable` int(5) NOT NULL DEFAULT '0' COMMENT 'GPU可分配数量',
  `cpu_total` int(5) NOT NULL DEFAULT '0' COMMENT 'CPU总数',
  `cpu_allocatable` int(5) NOT NULL DEFAULT '0' COMMENT 'CPU可分配数量',
  `mem_total` bigint(20) NOT NULL DEFAULT '0' COMMENT '内存总量(bytes)',
  `mem_allocatable` bigint(20) NOT NULL DEFAULT '0' COMMENT '内存可分配(bytes)',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted_at` datetime DEFAULT NULL COMMENT '删除时间',
  PRIMARY KEY (`id`),
  KEY `idx_cpod_id_deleted_at` (`cpod_id`,`deleted_at`),
  KEY `idx_cpod_id_node_name_deleted_at` (`cpod_id`,`node_name`,`deleted_at`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COMMENT='cpod节点信息表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_fileurl`
--

DROP TABLE IF EXISTS `sys_fileurl`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_fileurl` (
  `file_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `job_name` varchar(100) NOT NULL COMMENT 'job 名称',
  `file_url` varchar(355) DEFAULT NULL COMMENT '文件链接',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`file_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=123 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='模型文件表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_inference`
--

DROP TABLE IF EXISTS `sys_inference`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_inference` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `service_name` varchar(255) NOT NULL COMMENT '推理服务名',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `cpod_id` varchar(255) NOT NULL DEFAULT '' COMMENT 'cpod id',
  `status` tinyint(1) DEFAULT '0' COMMENT '状态：0等待部署、1部署中、2部署完成、3终止',
  `obtain_status` tinyint(1) DEFAULT '0' COMMENT '状态：1不需要下发、0需要下发',
  `billing_status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '账单状态（0 未结清、1 已结清）',
  `gpu_number` int(5) DEFAULT NULL COMMENT 'GPU数量',
  `gpu_type` varchar(60) DEFAULT NULL COMMENT 'GPU型号',
  `model_name` varchar(255) DEFAULT NULL COMMENT '模型名称',
  `model_id` varchar(255) DEFAULT NULL COMMENT 'modelstorage id',
  `model_size` bigint(20) DEFAULT NULL COMMENT '模型体积(字节)',
  `model_public` tinyint(4) DEFAULT NULL COMMENT '模型类型 1 公共 2 用户私有',
  `template` varchar(255) DEFAULT NULL COMMENT '推理模板',
  `url` varchar(512) NOT NULL DEFAULT '' COMMENT '服务的URL',
  `metadata` json DEFAULT NULL COMMENT '扩展字段',
  `start_time` datetime DEFAULT NULL COMMENT '推理服务启动时间',
  `end_time` datetime DEFAULT NULL COMMENT '推理服务终止时间',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_cpod_id` (`cpod_id`),
  KEY `idx_user_id` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=95 DEFAULT CHARSET=utf8mb4 COMMENT='推理服务表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_jupyterlab`
--

DROP TABLE IF EXISTS `sys_jupyterlab`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_jupyterlab` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `job_name` varchar(255) NOT NULL DEFAULT '' COMMENT '实例ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `cpod_id` varchar(255) NOT NULL DEFAULT '' COMMENT 'cpod id',
  `status` tinyint(4) NOT NULL DEFAULT '0' COMMENT '状态：0等待分配、1创建中、2运行中、3终止、4失败',
  `billing_status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '账单状态（0 未结清、1 已结清）',
  `instance_name` varchar(255) NOT NULL DEFAULT '' COMMENT '实例名称',
  `gpu_count` int(11) NOT NULL DEFAULT '0' COMMENT 'GPU数量',
  `gpu_prod` varchar(255) NOT NULL DEFAULT '' COMMENT 'GPU型号',
  `cpu_count` int(11) NOT NULL DEFAULT '1' COMMENT 'cpu数量',
  `mem_count` bigint(20) NOT NULL DEFAULT '0' COMMENT '内存 单位：字节',
  `data_volume_size` bigint(20) NOT NULL DEFAULT '0' COMMENT '数据卷 单位：字节',
  `model_id` varchar(255) NOT NULL DEFAULT '' COMMENT '挂载模型的id',
  `model_name` varchar(255) NOT NULL DEFAULT '' COMMENT '挂载模型的名字',
  `model_path` varchar(255) NOT NULL DEFAULT '' COMMENT '模型挂载路径',
  `resource` text NOT NULL COMMENT '挂载的资源',
  `replicas` tinyint(4) NOT NULL DEFAULT '1' COMMENT '副本数：0关闭、1运行',
  `url` varchar(512) NOT NULL DEFAULT '' COMMENT 'URL',
  `start_time` datetime DEFAULT NULL COMMENT '推理服务启动时间',
  `end_time` datetime DEFAULT NULL COMMENT '推理服务终止时间',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_job_name` (`job_name`),
  KEY `idx_user_id_status` (`user_id`,`status`),
  KEY `idx_user_id_gpu_prod` (`user_id`,`gpu_prod`)
) ENGINE=InnoDB AUTO_INCREMENT=94 DEFAULT CHARSET=utf8mb4 COMMENT='jupyterlab实例表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_order`
--

DROP TABLE IF EXISTS `sys_order`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_order` (
  `order_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `user_id` bigint(20) DEFAULT NULL COMMENT '用户ID',
  `job_name` varchar(100) DEFAULT NULL COMMENT '任务名称',
  `out_trade_no` varchar(56) DEFAULT NULL COMMENT '商户订单号',
  `body` varchar(366) DEFAULT NULL COMMENT '商品描述',
  `subject` varchar(180) DEFAULT NULL COMMENT '商品名称',
  `total_amount` double(10,2) DEFAULT NULL COMMENT '支付金额',
  `status` tinyint(4) DEFAULT '0' COMMENT '状态 0未付款 1已付款',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `trade_no` varchar(56) DEFAULT NULL COMMENT '支付宝交易号',
  PRIMARY KEY (`order_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=51 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='订单列表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_oss_resource`
--

DROP TABLE IF EXISTS `sys_oss_resource`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_oss_resource` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `resource_id` varchar(64) NOT NULL COMMENT '数据id',
  `resource_type` varchar(24) NOT NULL DEFAULT '' COMMENT '数据类型',
  `resource_name` varchar(255) NOT NULL DEFAULT '' COMMENT '缓存的数据名字',
  `resource_size` bigint(20) NOT NULL DEFAULT '0' COMMENT '资源体积(字节)',
  `public` tinyint(4) NOT NULL DEFAULT '1' COMMENT '资源是否公开 1 公共 2 用户私有',
  `user_id` varchar(255) NOT NULL COMMENT '用户ID',
  `meta` text NOT NULL COMMENT '扩展元数据',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_resource_id` (`resource_id`),
  KEY `idx_public_user_id` (`public`,`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=utf8mb4 COMMENT='oss资源表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_price`
--

DROP TABLE IF EXISTS `sys_price`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_price` (
  `price_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `gpu_prod` varchar(255) DEFAULT NULL COMMENT 'GPU型号',
  `amount` double(10,2) DEFAULT NULL COMMENT '基础价格/min/个',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  PRIMARY KEY (`price_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='单价表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_quota`
--

DROP TABLE IF EXISTS `sys_quota`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_quota` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `resource` varchar(255) NOT NULL DEFAULT '' COMMENT '资源类型',
  `quota` bigint(20) NOT NULL DEFAULT '0' COMMENT '资源配额',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COMMENT='用户配额表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_role`
--

DROP TABLE IF EXISTS `sys_role`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_role` (
  `role_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `name` varchar(100) NOT NULL COMMENT '名称',
  `level` int(50) DEFAULT NULL COMMENT '角色级别',
  `description` varchar(255) DEFAULT NULL COMMENT '描述',
  `data_scope` varchar(255) DEFAULT NULL COMMENT '数据权限',
  `create_by` varchar(255) DEFAULT NULL COMMENT '创建者',
  `update_by` varchar(255) DEFAULT NULL COMMENT '更新者',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  PRIMARY KEY (`role_id`) USING BTREE,
  UNIQUE KEY `uniq_name` (`name`),
  KEY `role_name_index` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='角色表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_roles_menus`
--

DROP TABLE IF EXISTS `sys_roles_menus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_roles_menus` (
  `menu_id` bigint(20) NOT NULL COMMENT '菜单ID',
  `role_id` bigint(20) NOT NULL COMMENT '角色ID',
  PRIMARY KEY (`menu_id`,`role_id`) USING BTREE,
  KEY `FKcngg2qadojhi3a651a5adkvbq` (`role_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='角色菜单关联';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_user`
--

DROP TABLE IF EXISTS `sys_user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_user` (
  `user_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `username` varchar(180) DEFAULT NULL COMMENT '用户名',
  `nick_name` varchar(255) DEFAULT NULL COMMENT '昵称',
  `gender` varchar(2) DEFAULT NULL COMMENT '性别',
  `phone` varchar(255) DEFAULT NULL COMMENT '手机号码',
  `email` varchar(180) DEFAULT NULL COMMENT '邮箱',
  `avatar_name` varchar(255) DEFAULT NULL COMMENT '头像地址',
  `avatar_path` varchar(255) DEFAULT NULL COMMENT '头像真实路径',
  `password` varchar(255) DEFAULT NULL COMMENT '密码',
  `is_admin` bit(1) DEFAULT b'0' COMMENT '是否为admin账号',
  `admin` tinyint(4) NOT NULL DEFAULT '0' COMMENT '管理员标志，0普通用户 1管理员 2超级管理员',
  `enabled` bigint(20) DEFAULT NULL COMMENT '状态：1启用、0禁用',
  `create_by` varchar(255) DEFAULT NULL COMMENT '创建者',
  `update_by` varchar(255) DEFAULT NULL COMMENT '更新者',
  `pwd_reset_time` datetime DEFAULT NULL COMMENT '修改密码的时间',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `user_type` int(5) NOT NULL DEFAULT '2' COMMENT '2为算力用户 3为算力源用户',
  `company_name` varchar(255) DEFAULT NULL COMMENT '公司名称',
  `company_phone` varchar(90) DEFAULT NULL COMMENT '公司联系方式',
  `company_other` varchar(255) DEFAULT NULL COMMENT '公司其他信息',
  `company_id` varchar(90) DEFAULT NULL COMMENT '算力源标签ID',
  PRIMARY KEY (`user_id`) USING BTREE,
  UNIQUE KEY `uniq_username` (`username`),
  UNIQUE KEY `uniq_email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=72 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='用户';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sys_user_job`
--

DROP TABLE IF EXISTS `sys_user_job`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sys_user_job` (
  `job_id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `cpod_id` varchar(60) DEFAULT NULL COMMENT 'cpod id',
  `work_status` tinyint(1) DEFAULT '0' COMMENT '状态：1失败、0运行、2完成',
  `obtain_status` tinyint(1) DEFAULT '0' COMMENT '状态：1不需要下发、0需要下发',
  `billing_status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '账单状态（0 未结清、1 已结清）',
  `job_name` varchar(255) DEFAULT NULL COMMENT '任务名称',
  `gpu_number` int(5) DEFAULT NULL COMMENT 'GPU数量',
  `gpu_type` varchar(60) DEFAULT NULL COMMENT 'GPU型号',
  `ckpt_path` varchar(255) DEFAULT NULL COMMENT 'cktp路径',
  `ckpt_vol` varchar(60) DEFAULT NULL COMMENT 'cktp容量',
  `model_path` varchar(255) DEFAULT NULL COMMENT 'save model路径',
  `model_vol` varchar(60) DEFAULT NULL COMMENT 'save model容量',
  `image_path` varchar(255) DEFAULT NULL COMMENT '镜像路径',
  `hf_url` varchar(255) DEFAULT NULL COMMENT 'HF公开训练数据URL',
  `dataset_path` varchar(255) DEFAULT NULL COMMENT '挂载路径',
  `job_type` varchar(60) DEFAULT NULL COMMENT '任务类型 mpi',
  `stop_type` tinyint(1) DEFAULT NULL COMMENT '0 自然终止 1设定时长',
  `stop_time` int(11) DEFAULT NULL COMMENT '设定时常以分钟为单位',
  `create_time` datetime DEFAULT NULL COMMENT '创建日期',
  `update_time` datetime DEFAULT NULL COMMENT '更新时间',
  `pretrained_model_name` varchar(255) DEFAULT NULL COMMENT '模型基座名称',
  `run_command` text COMMENT '模型启动命令',
  `callback_url` varchar(255) DEFAULT NULL COMMENT '第三方回调接口url',
  `pretrained_model_path` varchar(255) DEFAULT NULL COMMENT '模型基座路径',
  `dataset_name` varchar(255) DEFAULT NULL COMMENT '挂载路径名称',
  `json_all` mediumtext COMMENT 'json数据全包',
  `deleted` tinyint(1) DEFAULT '0' COMMENT '逻辑删除 0 未删除 1逻辑删除',
  PRIMARY KEY (`job_id`) USING BTREE,
  KEY `idx_job_name` (`job_name`(191)),
  KEY `idx_new_user_id` (`new_user_id`(191))
) ENGINE=InnoDB AUTO_INCREMENT=446 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='用户任务';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tool_alipay_config`
--

DROP TABLE IF EXISTS `tool_alipay_config`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tool_alipay_config` (
  `config_id` bigint(20) NOT NULL COMMENT 'ID',
  `app_id` varchar(255) DEFAULT NULL COMMENT '应用ID',
  `charset` varchar(255) DEFAULT NULL COMMENT '编码',
  `format` varchar(255) DEFAULT NULL COMMENT '类型 固定格式json',
  `gateway_url` varchar(255) DEFAULT NULL COMMENT '网关地址',
  `notify_url` varchar(255) DEFAULT NULL COMMENT '异步回调',
  `private_key` text COMMENT '私钥',
  `public_key` text COMMENT '公钥',
  `return_url` varchar(255) DEFAULT NULL COMMENT '回调地址',
  `sign_type` varchar(255) DEFAULT NULL COMMENT '签名方式',
  `sys_service_provider_id` varchar(255) DEFAULT NULL COMMENT '商户号',
  PRIMARY KEY (`config_id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='支付宝配置类';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user_balance`
--

DROP TABLE IF EXISTS `user_balance`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_balance` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `balance` decimal(12,2) NOT NULL DEFAULT '0.00' COMMENT '当前余额',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=47 DEFAULT CHARSET=utf8mb4 COMMENT='用户余额表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user_billing`
--

DROP TABLE IF EXISTS `user_billing`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_billing` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `billing_id` varchar(255) NOT NULL COMMENT '账单ID',
  `user_id` bigint(20) NOT NULL DEFAULT '0' COMMENT '用户ID',
  `new_user_id` varchar(255) NOT NULL DEFAULT '' COMMENT '用户ID',
  `amount` decimal(10,2) NOT NULL COMMENT '消费金额',
  `billing_status` tinyint(4) NOT NULL DEFAULT '0' COMMENT '账单状态（0 未支付、1 已支付、2 欠费）',
  `job_id` varchar(255) NOT NULL COMMENT '关联任务id',
  `job_type` varchar(50) NOT NULL COMMENT '关联任务类型（例如：finetune、inference）',
  `billing_time` datetime NOT NULL COMMENT '账单生成时间',
  `due_time` datetime DEFAULT NULL COMMENT '到期时间',
  `payment_time` datetime DEFAULT NULL COMMENT '支付时间',
  `description` text COMMENT '账单描述（可选，详细说明此次费用的具体内容）',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id_billing_status` (`user_id`,`billing_status`),
  KEY `idx_user_id_job_id` (`user_id`,`job_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_job_id` (`job_id`)
) ENGINE=InnoDB AUTO_INCREMENT=191876 DEFAULT CHARSET=utf8mb4 COMMENT='用户账单表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user_recharge`
--

DROP TABLE IF EXISTS `user_recharge`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_recharge` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `recharge_id` varchar(255) NOT NULL COMMENT '充值记录id',
  `user_id` varchar(255) NOT NULL COMMENT '用户ID',
  `amount` decimal(10,2) NOT NULL COMMENT '充值金额',
  `before_balance` decimal(10,2) NOT NULL COMMENT '充值前余额',
  `after_balance` decimal(10,2) NOT NULL COMMENT '充值后余额',
  `description` text NOT NULL COMMENT '描述',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted_at` datetime DEFAULT NULL COMMENT '删除时间',
  PRIMARY KEY (`id`),
  KEY `idx_recharge_id` (`recharge_id`,`deleted_at`),
  KEY `idx_user_id` (`user_id`,`deleted_at`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8mb4 COMMENT='用户充值记录表';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `verify_code`
--

DROP TABLE IF EXISTS `verify_code`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `verify_code` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `verify_key` varchar(255) NOT NULL DEFAULT '',
  `code` varchar(255) NOT NULL DEFAULT '' COMMENT '验证码',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_verify_key` (`verify_key`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COMMENT='验证码表';
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-08-08 11:11:59
