from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import os
import io
import uuid
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class DataProcessor:
    def __init__(self):
        self.df = None
        self.code_history = []
    
    def load_data(self, file_path):
        """加载Excel数据"""
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.df = pd.read_excel(file_path)
                code = f"""
# 数据加载代码
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# 加载数据
df = pd.read_excel('{file_path}')
print(f"数据形状: {self.df.shape}")
print(f"列名: {list(self.df.columns)}")
"""
                self.code_history.append(("数据加载", code))
                return True, f"成功加载数据，形状: {self.df.shape}"
            else:
                return False, "不支持的文件格式，请上传Excel文件"
        except Exception as e:
            return False, f"文件加载失败: {str(e)}"
    
    def handle_missing_values(self, method='drop', fill_value=None):
        """处理缺失值"""
        if self.df is None:
            return False, "请先上传数据文件", ""
        
        try:
            original_shape = self.df.shape
            
            if method == 'drop':
                self.df = self.df.dropna()
                code = """
# 缺失值处理 - 删除含有缺失值的行
df_cleaned = df.dropna()
print(f"原始数据形状: {original_shape}")
print(f"处理后数据形状: {df_cleaned.shape}")
"""
            elif method == 'fill_mean':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
                code = """
# 缺失值处理 - 用均值填充数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_cleaned = df.copy()
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
print(f"数值列: {list(numeric_cols)}")
print("用均值填充缺失值")
"""
            elif method == 'fill_median':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
                code = """
# 缺失值处理 - 用中位数填充数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_cleaned = df.copy()
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
print(f"数值列: {list(numeric_cols)}")
print("用中位数填充缺失值")
"""
            elif method == 'fill_value' and fill_value is not None:
                self.df = self.df.fillna(fill_value)
                code = f"""
# 缺失值处理 - 用指定值填充
df_cleaned = df.fillna({fill_value})
print(f"用值 {fill_value} 填充所有缺失值")
"""
            
            self.code_history.append(("缺失值处理", code))
            return True, f"缺失值处理完成。原始形状: {original_shape}, 处理后形状: {self.df.shape}", code
            
        except Exception as e:
            return False, f"缺失值处理失败: {str(e)}", ""
    
    def handle_outliers(self, method='iqr', threshold=3):
        """处理异常值"""
        if self.df is None:
            return False, "请先上传数据文件", ""
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            original_shape = self.df.shape
            
            if method == 'iqr':
                # IQR方法
                Q1 = self.df[numeric_cols].quantile(0.25)
                Q3 = self.df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = True
                for col in numeric_cols:
                    mask = mask & (self.df[col] >= lower_bound[col]) & (self.df[col] <= upper_bound[col])
                
                self.df = self.df[mask]
                
                code = """
# 异常值处理 - IQR方法
numeric_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 筛选出正常值
mask = True
for col in numeric_cols:
    mask = mask & (df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])

df_no_outliers = df[mask]
print(f"原始数据形状: {original_shape}")
print(f"处理后数据形状: {df_no_outliers.shape}")
"""
            
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
                mask = (z_scores < threshold).all(axis=1)
                self.df = self.df[mask]
                
                code = f"""
# 异常值处理 - Z-score方法 (阈值: {threshold})
from scipy import stats
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[numeric_cols]))
mask = (z_scores < {threshold}).all(axis=1)
df_no_outliers = df[mask]
print(f"原始数据形状: {original_shape}")
print(f"处理后数据形状: {df_no_outliers.shape}")
"""
            
            self.code_history.append(("异常值处理", code))
            return True, f"异常值处理完成。原始形状: {original_shape}, 处理后形状: {self.df.shape}", code
            
        except Exception as e:
            return False, f"异常值处理失败: {str(e)}", ""
    
    def handle_duplicates(self):
        """处理重复值"""
        if self.df is None:
            return False, "请先上传数据文件", ""
        
        try:
            original_shape = self.df.shape
            duplicates_count = self.df.duplicated().sum()
            self.df = self.df.drop_duplicates()
            
            code = """
# 重复值处理
print(f"发现重复行数: {duplicates_count}")
df_no_duplicates = df.drop_duplicates()
print(f"原始数据形状: {original_shape}")
print(f"处理后数据形状: {df_no_duplicates.shape}")
"""
            
            self.code_history.append(("重复值处理", code))
            return True, f"重复值处理完成。删除了 {duplicates_count} 行重复数据。原始形状: {original_shape}, 处理后形状: {self.df.shape}", code
            
        except Exception as e:
            return False, f"重复值处理失败: {str(e)}", ""
    
    def standardize_data(self, method='zscore', columns=None):
        """数据标准化"""
        if self.df is None:
            return False, "请先上传数据文件", ""
        
        try:
            if columns is None:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_cols = [col for col in columns if col in self.df.columns]
            
            if not numeric_cols:
                return False, "没有找到可标准化的数值列", ""
            
            if method == 'zscore':
                scaler = StandardScaler()
                self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
                
                code = """
# Z-score标准化
from sklearn.preprocessing import StandardScaler

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df_standardized[numeric_cols])

print(f"标准化的列: {numeric_cols}")
print("使用Z-score标准化: (x - μ) / σ")
"""
            
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
                
                code = """
# Min-Max标准化
from sklearn.preprocessing import MinMaxScaler

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df_standardized[numeric_cols])

print(f"标准化的列: {numeric_cols}")
print("使用Min-Max标准化: (x - min) / (max - min)")
"""
            
            self.code_history.append(("数据标准化", code))
            return True, f"数据标准化完成。标准化列: {numeric_cols}", code
            
        except Exception as e:
            return False, f"数据标准化失败: {str(e)}", ""
    
    def correlation_analysis(self):
        """相关性分析"""
        if self.df is None:
            return False, "请先上传数据文件", "", False
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return False, "需要至少2个数值列进行相关性分析", "", False
            
            correlation_matrix = self.df[numeric_cols].corr()
            
            code = """
# 相关性分析
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

print("相关性矩阵:")
print(correlation_matrix)

# 查找高相关性的特征对
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:  # 高相关性阈值
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  corr_value))

print("\\n高相关性特征对 (|r| > 0.7):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
"""
            
            self.code_history.append(("相关性分析", code))
            return True, f"相关性分析完成。分析了 {len(numeric_cols)} 个数值列", code, False
            
        except Exception as e:
            return False, f"相关性分析失败: {str(e)}", "", False
    
    def t_test(self, column1, column2=None, value=None):
        """t检验"""
        if self.df is None:
            return False, "请先上传数据文件", "", False
        
        try:
            if column1 not in self.df.columns:
                return False, f"列 '{column1}' 不存在", "", False
            
            if column2 is not None:
                # 双样本t检验
                if column2 not in self.df.columns:
                    return False, f"列 '{column2}' 不存在", "", False
                
                data1 = self.df[column1].dropna()
                data2 = self.df[column2].dropna()
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                code = f"""
# 双样本t检验
from scipy import stats

data1 = df['{column1}'].dropna()
data2 = df['{column2}'].dropna()

t_statistic, p_value = stats.ttest_ind(data1, data2)

print(f"双样本t检验结果:")
print(f"列1: {column1}, 样本数: {{len(data1)}}, 均值: {{data1.mean():.4f}}")
print(f"列2: {column2}, 样本数: {{len(data2)}}, 均值: {{data2.mean():.4f}}")
print(f"t统计量: {{t_statistic:.4f}}")
print(f"p值: {{p_value:.4f}}")
print(f"显著性水平0.05下{'拒绝' if p_value < 0.05 else '接受'}原假设")
"""
                
                result_text = f"双样本t检验: t统计量={t_stat:.4f}, p值={p_value:.4f}"
                
            elif value is not None:
                # 单样本t检验
                data = self.df[column1].dropna()
                t_stat, p_value = stats.ttest_1samp(data, value)
                
                code = f"""
# 单样本t检验
from scipy import stats

data = df['{column1}'].dropna()
test_value = {value}

t_statistic, p_value = stats.ttest_1samp(data, test_value)

print(f"单样本t检验结果:")
print(f"列: {column1}, 样本数: {{len(data)}}, 样本均值: {{data.mean():.4f}}")
print(f"检验值: {value}")
print(f"t统计量: {{t_statistic:.4f}}")
print(f"p值: {{p_value:.4f}}")
print(f"显著性水平0.05下{'拒绝' if p_value < 0.05 else '接受'}原假设")
"""
                
                result_text = f"单样本t检验: t统计量={t_stat:.4f}, p值={p_value:.4f}"
            
            else:
                return False, "请指定第二列或检验值", "", False
            
            self.code_history.append(("t检验", code))
            return True, result_text, code, False
            
        except Exception as e:
            return False, f"t检验失败: {str(e)}", "", False
    
    def chi_square_test(self, column1, column2):
        """卡方检验"""
        if self.df is None:
            return False, "请先上传数据文件", "", False
        
        try:
            if column1 not in self.df.columns or column2 not in self.df.columns:
                return False, "指定的列不存在", "", False
            
            # 创建列联表
            contingency_table = pd.crosstab(self.df[column1], self.df[column2])
            
            # 卡方检验
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            code = f"""
# 卡方检验
from scipy import stats
import pandas as pd

# 创建列联表
contingency_table = pd.crosstab(df['{column1}'], df['{column2}'])
print("列联表:")
print(contingency_table)

# 执行卡方检验
chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(contingency_table)

print(f"\\n卡方检验结果:")
print(f"卡方统计量: {{chi2_statistic:.4f}}")
print(f"p值: {{p_value:.4f}}")
print(f"自由度: {{degrees_of_freedom}}")
print(f"显著性水平0.05下{'拒绝' if p_value < 0.05 else '接受'}原假设(变量独立)")

print(f"\\n期望频率:")
print(expected_frequencies)
"""
            
            result_text = f"卡方检验: χ²={chi2:.4f}, p值={p_value:.4f}, 自由度={dof}"
            
            self.code_history.append(("卡方检验", code))
            return True, result_text, code, False
            
        except Exception as e:
            return False, f"卡方检验失败: {str(e)}", "", False
    
    def save_result(self, filename):
        """保存处理结果"""
        if self.df is None:
            return False, "没有数据可保存"
        
        try:
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            self.df.to_excel(filepath, index=False)
            return True, filepath
        except Exception as e:
            return False, f"保存失败: {str(e)}"
    
    def get_complete_code(self):
        """获取完整的Python代码"""
        if not self.code_history:
            return "# 没有执行任何操作"
        
        complete_code = "# 完整的数据预处理Python代码\n"
        complete_code += "# 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        
        for operation, code in self.code_history:
            complete_code += f"# {operation}\n"
            complete_code += code + "\n\n"
        
        return complete_code

# 全局数据处理器
processor = DataProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        if file and file.filename.endswith(('.xlsx', '.xls')):
            # 生成唯一文件名
            filename = str(uuid.uuid4()) + '_' + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 加载数据
            success, message = processor.load_data(filepath)
            
            if success:
                # 获取数据信息
                # 处理NaN值以避免JSON序列化问题
                sample_data = processor.df.head().fillna('null').to_dict('records')
                missing_values = processor.df.isnull().sum().to_dict()
                
                info = {
                    'shape': list(processor.df.shape),
                    'columns': list(processor.df.columns),
                    'dtypes': processor.df.dtypes.astype(str).to_dict(),
                    'missing_values': {k: int(v) for k, v in missing_values.items()},
                    'sample_data': sample_data
                }
                return jsonify({'success': True, 'message': message, 'data_info': info})
            else:
                return jsonify({'success': False, 'message': message})
        else:
            return jsonify({'success': False, 'message': '请上传Excel文件 (.xlsx 或 .xls)'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.json
        operation = data.get('operation')
        params = data.get('parameters', {})
        
        success = False
        message = ""
        code = ""
        can_download = True
        
        if operation == 'missing_values':
            method = params.get('method', 'drop')
            fill_value = params.get('fill_value')
            success, message, code = processor.handle_missing_values(method, fill_value)
        
        elif operation == 'outliers':
            method = params.get('method', 'iqr')
            threshold = params.get('threshold', 3)
            success, message, code = processor.handle_outliers(method, threshold)
        
        elif operation == 'duplicates':
            success, message, code = processor.handle_duplicates()
        
        elif operation == 'standardization':
            method = params.get('method', 'zscore')
            columns = params.get('columns')
            success, message, code = processor.standardize_data(method, columns)
        
        elif operation == 'correlation':
            success, message, code, can_download = processor.correlation_analysis()
        
        elif operation == 't_test':
            column1 = params.get('column1')
            column2 = params.get('column2')
            value = params.get('value')
            success, message, code, can_download = processor.t_test(column1, column2, value)
        
        elif operation == 'chi_square':
            column1 = params.get('column1')
            column2 = params.get('column2')
            success, message, code, can_download = processor.chi_square_test(column1, column2)
        
        else:
            message = "不支持的操作类型"
            can_download = False
        
        if not success and can_download:
            can_download = False
            message = "此种功能无法给出表格"
        
        # 获取完整代码
        complete_code = processor.get_complete_code()
        
        return jsonify({
            'success': success,
            'message': message,
            'code': code,
            'complete_code': complete_code,
            'can_download': can_download and success
        })
    
    except Exception as e:
        error_message = f"处理失败: {str(e)}"
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': error_message,
            'code': "",
            'complete_code': processor.get_complete_code(),
            'can_download': False
        })

@app.route('/download')
def download_result():
    try:
        filename = f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        success, result = processor.save_result(filename)
        
        if success:
            return send_file(result, as_attachment=True, download_name=filename)
        else:
            return jsonify({'success': False, 'message': result})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'下载失败: {str(e)}'})

@app.route('/reset', methods=['POST'])
def reset_processor():
    global processor
    processor = DataProcessor()
    return jsonify({'success': True, 'message': '已重置，可以上传新文件'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
