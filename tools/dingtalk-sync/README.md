# dingtalk-sync
同步钉钉部门和人员到数据库

## 使用方法
```bash
# 钉钉配置
export DINGTALK_APP_KEY=************
export DINGTALK_APP_SECRET=************

# 数据库配置
export DB_HOST=localhost      # 可选，默认localhost
export DB_PORT=3306          # 可选，默认3306
export DB_USER=root          # 可选，默认root
export DB_PASSWORD=******    # 必填
export DB_NAME=aiadmin      # 可选，默认aiadmin

python main.py
```

## 相关的数据表
```sql
CREATE TABLE `dingtalk_employee` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `union_id` varchar(64) NOT NULL DEFAULT '' COMMENT '钉钉用户唯一标识',
  `name` varchar(100) NOT NULL DEFAULT '' COMMENT '员工姓名',
  `department_id` int(11) NOT NULL DEFAULT '0' COMMENT '部门ID',
  `job_title` varchar(100) NOT NULL DEFAULT '' COMMENT '职位名称',
  `mobile` varchar(20) NOT NULL DEFAULT '' COMMENT '手机号码',
  `email` varchar(100) NOT NULL DEFAULT '' COMMENT '邮箱地址',
  `avatar_url` varchar(255) NOT NULL DEFAULT '' COMMENT '头像URL',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_union_id` (`union_id`),
  KEY `idx_department_id` (`department_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='钉钉员工信息表'

CREATE TABLE `dingtalk_department` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `department_id` int(11) NOT NULL DEFAULT '0' COMMENT '钉钉部门ID',
  `department_name` varchar(100) NOT NULL DEFAULT '' COMMENT '部门名称',
  `parent_department_id` int(11) NOT NULL DEFAULT '0' COMMENT '父部门ID',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_department_id` (`department_id`),
  KEY `idx_parent_department_id` (`parent_department_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='钉钉部门信息表'

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
  `from` varchar(50) DEFAULT 'cloud' COMMENT '注册来源',
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
) ENGINE=InnoDB AUTO_INCREMENT=75 DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPACT COMMENT='用户'
```

## main.py核心逻辑
1. 从DingTalk获取当前的全部组织架构和人员信息
2. 更新部门信息
  1. 如果没有该部门，插入记录到dingtalk_department
  2. 如果有该部门
    1. 部门信息有变更，更新记录
    2. 部门信息没有变更，跳过
3. 更新人员信息
  1. 如果没有该人员，插入信息到dingtalk_employee，并插入sys_user
  2. 如果有该人员
    1. 人员信息有变更，更新记录
    2. 人员信息没有更新，跳过
4. 移除人员或部门
  1. 如果部门不存在，移除部门记录
  2. 如果人员不存在，暂不处理。相当于会残留已离职的用户