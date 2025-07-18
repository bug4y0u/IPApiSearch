# IPApiSearch IP 归属地批量查询工具 (GUI)

一款基于 **Python + PyQt5** 的图形界面应用，面向 Excel 用户批量查询 IP 地址所属的国家 / 省份 / 城市，并写回原表。  
核心特点：

* 拖拽导入任意 Excel (`.xls` / `.xlsx`)  
* hyhdt.com **批量接口 + 多线程**，1000 IP ≈ 4–6 秒  
* 结果直接返回中文  
* 查询结果自动写入新列 `<原列名>_归属地`  
* SQLite 本地缓存，避免重复请求，可一键清理  
* 左右面板、表格列宽 **均可拖动调整**  
* 暗 / 亮主题自适应，全局异常提示、日志文件、可配置 API 凭证  
* 一行命令打包为单文件 EXE  

---

## 目录

1. [演示截图](#演示截图)  
2. [安装](#安装)  
3. [快速开始](#快速开始)  
4. [功能说明](#功能说明)  
5. [命令行打包](#命令行打包)  
6. [配置文件](#配置文件)  
7. [文件结构](#文件结构)  
8. [常见问题](#常见问题)  
9. [更新日志](#更新日志)  

---

## 演示截图
无

---

## 安装

```bash  
# 建议创建虚拟环境  
pip install PyQt5 pandas requests  
```

Python ≥ 3.8，Windows / macOS / Linux 通用。

---

## 快速开始

```bash  
python IPApiSearch.py  
```

* 首次启动会在程序同目录生成  
  `config.ini`   应用配置  
  `cache.db`    查询缓存  
  `logs/`       运行日志  

* 将含有 IP 的 Excel 直接拖进窗口，或「文件 → 打开 Excel」。

* 在左侧勾选含 IP 的列，点击「解析 IP 归属」。  
  进度条走完后右侧表格将多出 `xxx_归属地` 列，可直接导出。

---

## 功能说明

| 功能 | 说明 |
| ---- | ---- |
| 批量请求 | 使用 `https://www.hyhdt.com/api/getipaddress.ashx`，一次 ≤ 50 条，`ThreadPoolExecutor` 默认 6 线程 |
| 中文归属地 | 接口直接返回中文字段 |
| 列拖动 | 左右面板：`QSplitter`；表格列：`QHeaderView.Interactive` |
| 本地缓存 | 自动写入 / 读取 `cache.db`，避免重复查询；菜单可清理 30 天前记录 |
| 日志 | `logs/app.log`，捕获所有请求错误、异常栈 |
| 主题自适应 | 根据系统调色板自动套用深色样式 |
| 配置文件 | `config.ini`，使用 `RawConfigParser` 禁用 `%` 插值，粘贴任何 key 都 OK |

---

## 命令行打包

安装 PyInstaller：

```bash  
pip install pyinstaller  
```

生成单文件可执行（Windows 为例）：

```bash  
pyinstaller -F -w -i favicon.ico IPApiSearch.py  
```

首次运行时会在 EXE 所在目录生成 `config.ini / cache.db / logs/`。

---

## 配置文件

`config.ini` 示例

```ini  
[credential]  
appid  = ipsearch  
appkey = UzNu%2bPCKB%2fOx%2UyM%3d  
```

* `appkey` 在官方示例中已做百分号编码，**请勿再次 urlencode**。  
* 使用 `RawConfigParser`，`%` 无需转义。

---

## 文件结构

```  
├─ IPApiSearch.py         # 主程序（单文件）  
├─ favicon.ico             # 可选图标  
├─ cache.db                # 查询缓存（首次运行自动创建）  
├─ config.ini              # 配置文件（首次运行自动创建）  
└─ logs/  
   └─ app.log              # 运行日志  
```

---

## 常见问题

**Q1 : 打开后窗口空白？**  
A : 请确认只在 `_build_ui()` 里调用了一次 `setCentralWidget()`，并删除旧版布局代码。

**Q2 : 查询速度慢 / 超时？**  
A : 免费接口有并发 / 秒级速率限制。可在顶部常量 `MAX_WORKERS`、`BATCH_SIZE` 处调整，或使用自建 API。

**Q3 : 结果列为空？**  
A : 目标列含非法 IP 或接口返回失败即为空，可在日志查看原因。

---

## 更新日志

### v1.1.0 – 2025-07-16  
**切换接口 + 稳定性增强**

1. **接口替换**  
   • 从 `ip-api.com/batch` 切换至 `hyhdt.com/getipaddress.ashx`，一次 ≤ 50 条。  
   • 仅对 IPv6 做 URL-Encode；IPv4 原样。手工拼 URL，杜绝二次编码导致的 `code 300 参数错误`。

2. **合法 IP 过滤**  
   • 利用 `ipaddress` 标准库，仅保留合法 IPv4 / IPv6。  
   • Excel 列名、空白行等非 IP 再不会写入接口。

3. **线程安全退出**  
   • 新增 `MainWindow.closeEvent()`：解析线程未结束时阻止关闭窗口，彻底消灭 `QThread: Destroyed while thread is still running`。

4. **结果解析**  
   • 接口中的占位符 `-` 统一替换为空串。  
   • 新增 `area` 字段并拼接到最终地址。

5. **其它**  
   • 日志中记录完整请求 URL 与返回 JSON，方便排错。  
   • README、配置文件示例同步更新。

升级后示例输出：

```json  
{  
  "183.192.75.244": "中国 广东省 深圳市 南山区",  
  "39.188.218.96":  "中国 江苏省 南京市 建邺区" 
}  
```

---

Happy coding! 🚀