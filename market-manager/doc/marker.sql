DROP TABLE IF EXISTS `job_scheduler`;
CREATE TABLE `job_scheduler`
(
    `id`            BIGINT(11)    NOT NULL AUTO_INCREMENT,
    `job_id`        VARCHAR(45)   NOT NULL COMMENT '任务ID',
    `cpod_job_id`   VARCHAR(45)   NOT NULL COMMENT '调度到cpod的任务ID',
    `cpod_id`       VARCHAR(45)   NOT NULL COMMENT 'CpodID标识唯一cpodId',
    `state`         TINYINT(1)    NULL COMMENT '训练结果状态 0-完成 1-未完成',
    `job_url`       VARCHAR(255)  NULL COMMENT '训练结果地址',
    `create_at`     DATETIME      NULL,
    `update_at`     DATETIME      NULL,
    PRIMARY KEY (`id`),
    UNIQUE INDEX `idx_job_id`(`job_id`)
)
    ENGINE = InnoDB
    COMMENT = '任务调度表';

DROP TABLE IF EXISTS `cpod_resource`;
CREATE TABLE `cpod_resource`
(
    `id`            BIGINT(11)    NOT NULL AUTO_INCREMENT,
    `group_id`      VARCHAR(45)   NULL COMMENT 'Cpod 组标识唯一',
    `cpod_id`       VARCHAR(45)   NOT NULL COMMENT 'CpodID标识唯一cpodId',
    `gpu_total`     DOUBLE        NULL,
    `gpu_used`      DOUBLE        NULL,
    `gpu_free`      DOUBLE        NULL,
    `create_at`     DATETIME      NULL,
    `update_at`     DATETIME      NULL,
    PRIMARY KEY (`id`),
    UNIQUE INDEX `idx_cpod_id`(`cpod_id`)
)
    ENGINE = InnoDB
    COMMENT = 'cpod资源表';






