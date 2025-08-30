// 全局变量
let currentDataInfo = null;

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

// 初始化事件监听器
function initializeEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    // 文件拖拽事件
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // 文件选择事件
    fileInput.addEventListener('change', handleFileSelect);
    
    // 表单联动事件
    document.getElementById('missingMethod').addEventListener('change', toggleFillValue);
    document.getElementById('outlierMethod').addEventListener('change', toggleThreshold);
    document.getElementById('tTestType').addEventListener('change', toggleTTestInputs);
}

// 拖拽处理函数
function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 文件处理函数
function handleFile(file) {
    // 验证文件类型
    if (!file.name.match(/\.(xlsx|xls)$/)) {
        showAlert('请选择Excel文件 (.xlsx 或 .xls)', 'danger');
        return;
    }
    
    // 验证文件大小 (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('文件大小不能超过16MB', 'danger');
        return;
    }
    
    uploadFile(file);
}

// 上传文件
function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            currentDataInfo = data.data_info;
            displayDataInfo(data.data_info);
            populateColumnSelects(data.data_info.columns);
            showSection('data-info-section');
            showSection('functions-section');
            hideSection('upload-section');
            showAlert(data.message, 'success');
        } else {
            showAlert(data.message, 'danger');
        }
    })
    .catch(error => {
        showLoading(false);
        showAlert('上传失败：' + error.message, 'danger');
    });
}

// 显示数据信息
function displayDataInfo(dataInfo) {
    const dataInfoDiv = document.getElementById('dataInfo');
    
    let html = `
        <div class="row">
            <div class="col-md-6">
                <h6><i class="fas fa-table"></i> 基本信息</h6>
                <p><strong>数据形状：</strong> ${dataInfo.shape[0]} 行 × ${dataInfo.shape[1]} 列</p>
                <p><strong>列名：</strong> ${dataInfo.columns.join(', ')}</p>
            </div>
            <div class="col-md-6">
                <h6><i class="fas fa-exclamation-circle"></i> 缺失值统计</h6>
    `;
    
    const missingValues = dataInfo.missing_values;
    let hasMissing = false;
    for (const [col, count] of Object.entries(missingValues)) {
        if (count > 0) {
            html += `<p><strong>${col}：</strong> ${count} 个缺失值</p>`;
            hasMissing = true;
        }
    }
    
    if (!hasMissing) {
        html += '<p class="text-success">无缺失值</p>';
    }
    
    html += `
            </div>
        </div>
        <div class="mt-3">
            <h6><i class="fas fa-eye"></i> 数据预览 (前5行)</h6>
            <div class="table-responsive">
                <table class="table table-striped table-sm">
                    <thead>
                        <tr>
    `;
    
    // 表头
    dataInfo.columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // 数据行
    dataInfo.sample_data.forEach(row => {
        html += '<tr>';
        dataInfo.columns.forEach(col => {
            const value = row[col];
            html += `<td>${value !== null && value !== undefined ? value : '<span class="text-muted">null</span>'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table></div></div>';
    
    dataInfoDiv.innerHTML = html;
}

// 填充列选择下拉框
function populateColumnSelects(columns) {
    const selects = [
        'tTestColumn1', 'tTestColumn2', 
        'chiSquareColumn1', 'chiSquareColumn2'
    ];
    
    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        select.innerHTML = '<option value="">选择列</option>';
        
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        });
    });
}

// 表单联动函数
function toggleFillValue() {
    const method = document.getElementById('missingMethod').value;
    const fillValueDiv = document.getElementById('fillValueDiv');
    fillValueDiv.style.display = method === 'fill_value' ? 'block' : 'none';
}

function toggleThreshold() {
    const method = document.getElementById('outlierMethod').value;
    const thresholdDiv = document.getElementById('thresholdDiv');
    thresholdDiv.style.display = method === 'zscore' ? 'block' : 'none';
}

function toggleTTestInputs() {
    const testType = document.getElementById('tTestType').value;
    const column2Div = document.getElementById('tTestColumn2Div');
    const valueDiv = document.getElementById('tTestValueDiv');
    
    if (testType === 'two_sample') {
        column2Div.style.display = 'block';
        valueDiv.style.display = 'none';
    } else {
        column2Div.style.display = 'none';
        valueDiv.style.display = 'block';
    }
}

// 数据处理函数
function processMissingValues() {
    const method = document.getElementById('missingMethod').value;
    const fillValue = document.getElementById('fillValue').value;
    
    const params = { method };
    if (method === 'fill_value' && fillValue !== '') {
        params.fill_value = parseFloat(fillValue);
    }
    
    processData('missing_values', params);
}

function processOutliers() {
    const method = document.getElementById('outlierMethod').value;
    const threshold = document.getElementById('zThreshold').value;
    
    const params = { method };
    if (method === 'zscore') {
        params.threshold = parseFloat(threshold);
    }
    
    processData('outliers', params);
}

function processDuplicates() {
    processData('duplicates', {});
}

function processStandardization() {
    const method = document.getElementById('standardizationMethod').value;
    processData('standardization', { method });
}

function processCorrelation() {
    processData('correlation', {});
}

function processTTest() {
    const column1 = document.getElementById('tTestColumn1').value;
    const testType = document.getElementById('tTestType').value;
    
    if (!column1) {
        showAlert('请选择第一列', 'warning');
        return;
    }
    
    const params = { column1 };
    
    if (testType === 'two_sample') {
        const column2 = document.getElementById('tTestColumn2').value;
        if (!column2) {
            showAlert('请选择第二列', 'warning');
            return;
        }
        params.column2 = column2;
    } else {
        const value = document.getElementById('tTestValue').value;
        if (value === '') {
            showAlert('请输入检验值', 'warning');
            return;
        }
        params.value = parseFloat(value);
    }
    
    processData('t_test', params);
}

function processChiSquare() {
    const column1 = document.getElementById('chiSquareColumn1').value;
    const column2 = document.getElementById('chiSquareColumn2').value;
    
    if (!column1 || !column2) {
        showAlert('请选择两列进行卡方检验', 'warning');
        return;
    }
    
    processData('chi_square', { column1, column2 });
}

// 通用数据处理函数
function processData(operation, parameters) {
    showLoading(true);
    hideSection('result-section');
    
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            operation: operation,
            parameters: parameters
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        displayResult(data);
    })
    .catch(error => {
        showLoading(false);
        showAlert('处理失败：' + error.message, 'danger');
    });
}

// 显示处理结果
function displayResult(data) {
    const resultSection = document.getElementById('result-section');
    const resultMessage = document.getElementById('resultMessage');
    const pythonCode = document.getElementById('pythonCode');
    const downloadSection = document.getElementById('downloadSection');
    
    // 显示结果消息
    const messageClass = data.success ? 'result-success' : 'result-error';
    resultMessage.innerHTML = `<div class="${messageClass}">${data.message}</div>`;
    
    // 显示Python代码
    pythonCode.textContent = data.complete_code;
    
    // 控制下载按钮显示
    downloadSection.style.display = data.can_download ? 'block' : 'none';
    
    // 显示结果区域
    showSection('result-section');
    
    // 滚动到结果区域
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// 下载结果文件
function downloadResult() {
    window.location.href = '/download';
}

// 复制代码到剪贴板
function copyCode() {
    const codeElement = document.getElementById('pythonCode');
    const textArea = document.createElement('textarea');
    textArea.value = codeElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    showAlert('代码已复制到剪贴板', 'success');
}

// 重置处理器
function resetProcessor() {
    fetch('/reset', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 重置页面状态
            currentDataInfo = null;
            hideSection('data-info-section');
            hideSection('functions-section');
            hideSection('result-section');
            showSection('upload-section');
            
            // 清空文件输入
            document.getElementById('fileInput').value = '';
            
            showAlert(data.message, 'success');
        }
    })
    .catch(error => {
        showAlert('重置失败：' + error.message, 'danger');
    });
}

// 工具函数
function showSection(sectionId) {
    document.getElementById(sectionId).classList.remove('hidden');
}

function hideSection(sectionId) {
    document.getElementById(sectionId).classList.add('hidden');
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    loading.style.display = show ? 'block' : 'none';
}

function showAlert(message, type) {
    // 创建临时提示框
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 3秒后自动移除
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 3000);
}
