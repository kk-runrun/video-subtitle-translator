# 视频字幕提取工作站（Streamlit）

## 1. 环境准备（Python 3.8）

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

如果你要启用 EasyOCR 的 GPU，请先安装与你环境匹配的 PyTorch CUDA 版本（示例 cu118）：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

可选增强（英文纠错）：

```powershell
python -m pip install pyenchant
```

## 2. 运行

```powershell
python -m streamlit run app.py
```

## 3. 页面中你要填的位置

左侧栏支持三种模式：

- `单视频`：填写 `本地视频绝对路径`，例如 `D:\视频\test.mp4`
- `批处理`：
  - 直接拖多个视频到页面
  - 或在 `本地文件/文件夹绝对路径（每行一个，可混填）` 中填写多个视频路径 / 文件夹路径
  - 文件夹会递归扫描视频文件
- `在线链接`：
  - 在 `在线视频链接（每行一个）` 中填写多个可直接下载的视频链接
  - 系统会先下载到本地临时缓存，再复用原有批处理识别流程

然后点 `开始处理`。
处理中可用：`暂停`、`继续处理`、`结束处理（重置）`。

批处理和在线链接模式默认最多一次 `200` 个视频，超出请分批执行。

## 4. 识别规则

- 识别视频下方多区域（ROI）：
- 底部灰底区域和白字区域分别采用不同识别策略
- 白字支持 5-10 帧投票融合（频率 + 置信度）
- 同一行的紧凑文字会自动拼接为一句
- 上下两行会分成两句分别输出
- 零散字符/低置信度文本会过滤，不写入结果
- 当前模型聚焦英文与常见符号（如 `&`、`%`、`-`、`/` 等）

## 5. 输出

单视频模式页面实时表格两列：
- 视频名称
- 视频原字幕

支持导出：
- CSV
- Excel

批处理模式会在本地生成一个输出文件夹，包含：
- 每个视频各自的 CSV
- 每个视频各自的 Excel
- 全部视频汇总 CSV / Excel
- 一份可直接点击下载的 ZIP 压缩包
