# 数据预处理在线工具

一个功能强大的数据预处理Web应用，支持Excel文件的在线处理和分析。

## ✨ 主要功能

### 📊 数据清洗
- **缺失值处理**
  - 删除含缺失值的行
  - 用均值填充数值列
  - 用中位数填充数值列
  - 用指定值填充

- **异常值处理**
  - IQR方法检测和移除异常值
  - Z-score方法检测和移除异常值（可调阈值）

- **重复值处理**
  - 自动检测并删除重复行

### 📏 数据标准化
- **Z-score标准化**：(x - μ) / σ
- **Min-Max标准化**：(x - min) / (max - min)

### 📈 统计分析
- **相关性分析**：计算数值列之间的相关系数矩阵
- **t检验**：
  - 单样本t检验
  - 双样本t检验
- **卡方检验**：测试两个分类变量之间的独立性

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务器
```bash
python run.py
```

### 3. 访问应用
- 本地访问：http://localhost:5000
- 局域网访问：http://你的IP地址:5000

## 💻 使用方法

1. **上传数据**
   - 拖拽Excel文件到上传区域
   - 或点击选择文件按钮
   - 支持.xlsx和.xls格式，最大16MB

2. **查看数据信息**
   - 数据形状和基本统计
   - 缺失值统计
   - 数据预览

3. **选择处理功能**
   - 从数据清洗、标准化、统计分析三个类别中选择
   - 根据需要配置参数

4. **获取结果**
   - 下载处理后的Excel文件
   - 复制完整的Python代码
   - 查看处理结果和统计信息

## 📁 项目结构

```
数据预处理网站/
│
├── app.py                 # Flask主应用
├── run.py                 # 启动脚本
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
│
├── templates/            # HTML模板
│   └── index.html        # 主页面
│
├── static/              # 静态文件
│   └── js/
│       └── main.js      # 前端JavaScript
│
├── uploads/             # 上传文件目录（自动创建）
└── results/             # 结果文件目录（自动创建）
```

## 🛠️ 技术栈

### 后端
- **Flask**：Web框架
- **pandas**：数据处理
- **numpy**：数值计算
- **scikit-learn**：数据预处理和标准化
- **scipy**：统计分析
- **openpyxl**：Excel文件处理

### 前端
- **Bootstrap 5**：UI框架
- **Font Awesome**：图标库
- **vanilla JavaScript**：交互逻辑

## 📝 API接口

### 文件上传
- `POST /upload`
- 接受Excel文件，返回数据基本信息

### 数据处理
- `POST /process`
- 执行指定的数据处理操作

### 文件下载
- `GET /download`
- 下载处理后的Excel文件

### 重置
- `POST /reset`
- 重置处理器状态

## ⚙️ 配置说明

### 文件大小限制
在`app.py`中修改：
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### 端口配置
在`run.py`中修改：
```python
app.run(host='0.0.0.0', port=5000)
```

## 🔧 部署说明

### 本地部署
直接运行`python run.py`即可

### 生产环境部署
推荐使用Gunicorn：
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker部署
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "run.py"]
```

## 📊 支持的数据格式

- **Excel文件**：.xlsx, .xls
- **数据要求**：
  - 第一行应为列标题
  - 数值列用于数值处理和统计分析
  - 分类列用于卡方检验

## ⚠️ 注意事项

1. **内存使用**：大文件处理可能消耗较多内存
2. **数据隐私**：上传的文件会临时存储在服务器上
3. **浏览器兼容**：推荐使用现代浏览器（Chrome、Firefox、Safari、Edge）
4. **网络要求**：需要加载Bootstrap和Font Awesome的CDN资源

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请创建Issue或联系开发者。

---

**享受数据预处理的乐趣！** 🎉
