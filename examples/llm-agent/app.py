from flask import Flask, request, jsonify
import sys , os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agent_core import process_req , get_service_list , register_tool

app = Flask(__name__)

# 将process_req函数封装成一个视图
@app.route('/process', methods=['POST'])
def process_request():
    req = request.json.get('request')
    res = process_req(req)
    return jsonify({'response': res})

# 获取服务列表
@app.route('/services', methods=['GET'])
def get_services():
    services = get_service_list()  # 实现获取服务列表的逻辑
    return jsonify({'services': services})

# 注册工具
@app.route('/register', methods=['POST'])
def register():
    # 注册工具的逻辑
    register_tool()
    return jsonify({'message': 'Tool registered successfully'})

# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
