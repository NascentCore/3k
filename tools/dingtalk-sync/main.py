import os
import pymysql
import requests
import uuid
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DingTalkSync:
    def __init__(self):
        try:
            self.app_key = os.environ.get('DINGTALK_APP_KEY')
            self.app_secret = os.environ.get('DINGTALK_APP_SECRET')
            if not self.app_key or not self.app_secret:
                raise ValueError("请设置DINGTALK_APP_KEY和DINGTALK_APP_SECRET环境变量")
            
            self.access_token = self._get_access_token()
            self.db = self._connect_db()
            logger.info("初始化成功")
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def _execute_sql(self, cursor, sql, params=None):
        """执行SQL并处理错误"""
        try:
            cursor.execute(sql, params)
        except pymysql.Error as e:
            logger.error(f"SQL执行失败: {sql}")
            logger.error(f"参数: {params}")
            logger.error(f"错误信息: {str(e)}")
            raise

    def _get_access_token(self):
        """获取钉钉访问令牌"""
        url = f"https://oapi.dingtalk.com/gettoken?appkey={self.app_key}&appsecret={self.app_secret}"
        response = requests.get(url)
        return response.json().get("access_token")
    
    def _connect_db(self):
        """连接数据库"""
        host = os.environ.get('DB_HOST', 'localhost')
        port = int(os.environ.get('DB_PORT', '3306'))
        user = os.environ.get('DB_USER', 'root')
        password = os.environ.get('DB_PASSWORD')
        database = os.environ.get('DB_NAME', 'aiadmin')
        
        if not password:
            raise ValueError("请设置DB_PASSWORD环境变量")
        
        return pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
    
    def get_all_departments(self):
        """获取所有部门信息"""
        url = f"https://oapi.dingtalk.com/department/list?access_token={self.access_token}"
        response = requests.get(url)
        return response.json().get("department", [])
    
    def get_department_users(self, dept_id):
        """获取部门下的所有用户"""
        url = f"https://oapi.dingtalk.com/topapi/v2/user/list?access_token={self.access_token}"
        data = {
            "dept_id": dept_id,
            "cursor": 0,
            "size": 100
        }
        response = requests.post(url, json=data)
        result = response.json()
        
        all_users = []
        while True:
            if not result.get('result'):
                break
                
            users = result.get('result', {}).get('list', [])
            all_users.extend(users)
            
            next_cursor = result.get('result', {}).get('next_cursor', 0)
            if not next_cursor:
                break
                
            data['cursor'] = next_cursor
            response = requests.post(url, json=data)
            result = response.json()
        
        return all_users
    
    def sync_departments(self, departments):
        """同步部门信息"""
        cursor = self.db.cursor()
        try:
            # 获取现有部门ID列表
            self._execute_sql(cursor, "SELECT department_id FROM dingtalk_department")
            existing_dept_ids = {row[0] for row in cursor.fetchall()}
            
            # 更新或插入部门信息
            for dept in departments:
                dept_id = dept['id']
                try:
                    if dept_id in existing_dept_ids:
                        # 更新现有部门
                        sql = """
                        UPDATE dingtalk_department 
                        SET department_name = %s, parent_department_id = %s 
                        WHERE department_id = %s
                        """
                        self._execute_sql(cursor, sql, (dept['name'], dept.get('parentid', 0), dept_id))
                        logger.info(f"更新部门: {dept['name']}({dept_id})")
                    else:
                        # 插入新部门
                        sql = """
                        INSERT INTO dingtalk_department 
                        (department_id, department_name, parent_department_id) 
                        VALUES (%s, %s, %s)
                        """
                        self._execute_sql(cursor, sql, (dept_id, dept['name'], dept.get('parentid', 0)))
                        logger.info(f"新增部门: {dept['name']}({dept_id})")
                    
                    existing_dept_ids.discard(dept_id)
                except Exception as e:
                    logger.error(f"处理部门 {dept['name']}({dept_id}) 时出错: {str(e)}")
                    continue
            
            # 删除不存在的部门
            if existing_dept_ids:
                try:
                    self._execute_sql(
                        cursor,
                        "DELETE FROM dingtalk_department WHERE department_id IN %s",
                        (tuple(existing_dept_ids),)
                    )
                    logger.info(f"删除部门: {existing_dept_ids}")
                except Exception as e:
                    logger.error(f"删除旧部门时出错: {str(e)}")
            
            self.db.commit()
            logger.info("部门同步完成")
        except Exception as e:
            self.db.rollback()
            logger.error(f"部门同步失败: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def sync_employees(self):
        """同步员工信息"""
        cursor = self.db.cursor()
        try:
            # 获取现有员工union_id列表
            self._execute_sql(cursor, "SELECT union_id FROM dingtalk_employee")
            existing_union_ids = {row[0] for row in cursor.fetchall()}
            
            # 遍历所有部门获取员工
            departments = self.get_all_departments()
            for dept in departments:
                try:
                    users = self.get_department_users(dept['id'])
                    logger.info(f"获取到部门 {dept['name']}({dept['id']}) 的 {len(users)} 个用户")
                    
                    for user in users:
                        try:
                            union_id = user.get('unionid')
                            if not union_id:
                                logger.warning(f"用户 {user.get('name')} 没有unionid，跳过")
                                continue
                                
                            if union_id in existing_union_ids:
                                # 更新现有员工
                                self._update_employee(cursor, user, dept['id'], union_id)
                                logger.info(f"更新员工: {user['name']}({union_id})")
                            else:
                                # 插入新员工
                                self._insert_employee(cursor, user, dept['id'], union_id)
                                logger.info(f"新增员工: {user['name']}({union_id})")
                            
                            existing_union_ids.discard(union_id)
                        except Exception as e:
                            logger.error(f"处理用户 {user.get('name')} 时出错: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"处理部门 {dept['name']} 的用户时出错: {str(e)}")
                    continue
            
            self.db.commit()
            logger.info("员工同步完成")
        except Exception as e:
            self.db.rollback()
            logger.error(f"员工同步失败: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def _update_employee(self, cursor, user, dept_id, union_id):
        """更新员工信息"""
        sql = """
        UPDATE dingtalk_employee 
        SET name = %s, department_id = %s, job_title = %s,
            mobile = %s, email = %s, avatar_url = %s
        WHERE union_id = %s
        """
        self._execute_sql(cursor, sql, (
            user['name'], dept_id, user.get('position', ''),
            user.get('mobile', ''), user.get('email', ''),
            user.get('avatar', ''), union_id
        ))

    def _insert_employee(self, cursor, user, dept_id, union_id):
        """插入新员工"""
        # 插入dingtalk_employee表
        sql = """
        INSERT INTO dingtalk_employee 
        (union_id, name, department_id, job_title, mobile, email, avatar_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._execute_sql(cursor, sql, (
            union_id, user['name'], dept_id, user.get('position', ''),
            user.get('mobile', ''), user.get('email', ''),
            user.get('avatar', '')
        ))
        
        # 创建系统用户
        self._create_sys_user(user)

    def _create_sys_user(self, dingtalk_user):
        """创建系统用户"""
        cursor = self.db.cursor()
        
        # 检查用户是否已存在
        cursor.execute(
            "SELECT user_id FROM sys_user WHERE new_user_id = %s",
            (dingtalk_user.get('unionid', ''),)
        )
        if cursor.fetchone():
            return
        
        # 创建新系统用户
        sql = """
        INSERT INTO sys_user 
        (username, nick_name, gender, phone, email, avatar_path, 
         enabled, `from`, create_time, user_type, new_user_id)
        VALUES (%s, %s, %s, %s, %s, %s, 1, 'dingtalk', %s, 2, %s)
        """
        cursor.execute(sql, (
            dingtalk_user['name'],
            dingtalk_user['name'],
            '1' if dingtalk_user.get('gender', 'male') == 'male' else '2',
            dingtalk_user.get('mobile', ''),
            dingtalk_user.get('email', '') or f"{uuid.uuid4().hex[:8]}@sxwl.ai",
            dingtalk_user.get('avatar', ''),
            datetime.now(),
            dingtalk_user.get('unionid', '')
        ))
        
        self.db.commit()
        cursor.close()
    
    def sync(self):
        """执行完整的同步流程"""
        try:
            departments = self.get_all_departments()
            logger.info(f"获取到 {len(departments)} 个部门")
            
            self.sync_departments(departments)
            self.sync_employees()
            
            logger.info("同步完成")
        except Exception as e:
            logger.error(f"同步失败: {str(e)}")
        finally:
            self.db.close()

if __name__ == "__main__":
    syncer = DingTalkSync()
    syncer.sync()
