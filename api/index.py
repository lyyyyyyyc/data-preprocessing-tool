import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel需要这个函数作为入口点
def handler(request):
    return app(request.environ, lambda status, headers: None)

# 导出app供Vercel使用
application = app

if __name__ == "__main__":
    app.run()
