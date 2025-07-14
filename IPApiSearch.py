#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP 归属地批量查询工具 – 并发+可拖动版
author : YourName
license: MIT
"""

from __future__ import annotations
import sys, time, logging, threading, traceback, sqlite3, configparser
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from pandas import DataFrame

from PyQt5.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QObject,
    QThread, pyqtSignal
)
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QListWidget, QPushButton, QFileDialog, QTableView, QHeaderView,
    QProgressBar, QMessageBox, QAction, QMenu, QLabel, QSplitter
)

# ─────────────── 路径 & 日志 ───────────────
IS_FROZEN = getattr(sys, "frozen", False)
RSC_DIR   = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
RUN_DIR   = Path(sys.executable).parent if IS_FROZEN else Path(__file__).resolve().parent

CONFIG_FILE = RUN_DIR / "config.ini"
CACHE_FILE  = RUN_DIR / "cache.db"
LOG_DIR     = RUN_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    encoding="utf-8"
)
log = logging.getLogger("ip-locator")

# ─────────────── 配置文件 ───────────────
def init_config() -> dict:
    """
    1. 程序启动即保证 config.ini 存在
    2. 若文件损坏自动备份并重建
    3. 关闭 % 插值 (RawConfigParser)
    """
    default = {"credential": {"appid": "ipsearch", "appkey": ""}}
    cp = configparser.RawConfigParser()
    cp.read_dict(default)

    try:
        if CONFIG_FILE.exists():
            cp.read(CONFIG_FILE, encoding="utf-8")
        else:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                cp.write(f)
            log.info("首次启动已生成默认 config.ini")
    except (configparser.Error, UnicodeDecodeError) as e:
        bak = CONFIG_FILE.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}")
        CONFIG_FILE.rename(bak)
        log.error("config.ini 损坏已备份为 %s: %s", bak.name, e)
        cp = configparser.RawConfigParser()
        cp.read_dict(default)
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            cp.write(f)

    return {s: dict(cp.items(s)) for s in cp.sections()}

CFG = init_config()

# ─────────────── SQLite 缓存 ───────────────
class CacheDB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self._init()

    def _init(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS ip_cache(
                              ip TEXT PRIMARY KEY,
                              country TEXT, region TEXT, city TEXT,
                              ts INTEGER)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ts ON ip_cache(ts)")
            self.conn.commit()

    def fetch(self, ips: List[str]) -> Dict[str, Dict]:
        if not ips:
            return {}
        q = "SELECT ip,country,region,city FROM ip_cache WHERE ip IN (%s)" % (
            ",".join("?" * len(ips))
        )
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(q, ips)
            return {ip: dict(country=c, region=r, city=ci)
                    for ip, c, r, ci in cur.fetchall()}

    def save(self, mapping: Dict[str, Dict]):
        if not mapping:
            return
        rows = [(ip,
                 d.get("country", ""),
                 d.get("region", ""),
                 d.get("city", ""),
                 int(time.time())) for ip, d in mapping.items()]
        with self.lock:
            self.conn.executemany("REPLACE INTO ip_cache VALUES(?,?,?,?,?)", rows)
            self.conn.commit()

    def cleanup(self, days: int = 30):
        limit = int(time.time() - days * 86400)
        with self.lock:
            self.conn.execute("DELETE FROM ip_cache WHERE ts<?", (limit,))
            self.conn.commit()

# ─────────────── Pandas -> QtModel ───────────────
class PandasModel(QAbstractTableModel):
    def __init__(self, df: DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        val = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(val) else str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[section] if orientation == Qt.Horizontal else section)

    def setDataFrame(self, df: DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

# ─────────────── 并发解析 Worker ───────────────
BATCH_SIZE  = 100   # ip-api.com 限制
MAX_WORKERS = 6

class ParseWorker(QObject):
    progress = pyqtSignal(int)   # 已完成数量
    finished = pyqtSignal(dict)  # {ip: {...}}

    def __init__(self, ips: List[str]):
        super().__init__()
        self._ips = ips
        self._session = requests.Session()
        self._done = 0
        self._lock = threading.Lock()

    def run(self):
        out: Dict[str, Dict] = {}
        batches = [self._ips[i:i+BATCH_SIZE] for i in range(0, len(self._ips), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future2batch = {pool.submit(self._query_batch, b): b for b in batches}
            for fut in as_completed(future2batch):
                batch_ips = future2batch[fut]
                try:
                    out.update(fut.result())
                except Exception as e:
                    log.error("批次 %s 解析失败：%s", batch_ips, e)
                    out.update({ip: dict(country="", region="", city="") for ip in batch_ips})
                with self._lock:
                    self._done += len(batch_ips)
                    self.progress.emit(self._done)
        self.finished.emit(out)

    def _query_batch(self, ip_list: List[str]) -> Dict[str, Dict]:
        url = "http://ip-api.com/batch"
        params = {
            "fields": "status,country,regionName,city",
             "lang":   "zh-CN"          # ← 新增
        }
        r = self._session.post(url, json=ip_list, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        mapping = {}
        for ip, rec in zip(ip_list, data):
            if rec.get("status") == "success":
                mapping[ip] = dict(country=rec.get("country", ""),
                                   region=rec.get("regionName", ""),
                                   city=rec.get("city", ""))
            else:
                mapping[ip] = dict(country="", region="", city="")
        return mapping

# ─────────────── 主窗口 ───────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IP 归属地批量查询工具")
        self.resize(1000, 650)
        if (RSC_DIR / "favicon.ico").exists():
            self.setWindowIcon(QIcon(str(RSC_DIR / "favicon.ico")))

        self.df: DataFrame = pd.DataFrame()
        self.cache = CacheDB(CACHE_FILE)

        self._build_ui()
        self._build_menu()

        self.pbar = QProgressBar()
        self.statusBar().addPermanentWidget(self.pbar)
        self.pbar.hide()

        self.setAcceptDrops(True)

    # ---------- UI ----------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)      # 唯一一次 setCentralWidget

        # ---- 左侧列选择 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.list_cols = QListWidget()
        self.list_cols.setSelectionMode(QListWidget.MultiSelection)
        btn_parse = QPushButton("解析 IP 归属")
        btn_parse.clicked.connect(self.parse_selected)

        left_layout.addWidget(QLabel("列选择"))
        left_layout.addWidget(self.list_cols, 1)
        left_layout.addWidget(btn_parse)

        splitter.addWidget(left_widget)

        # ---- 右侧表格 ----
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # ← 关键
        header.setSectionsMovable(False)       # 只改宽度，不允许交换列（可选）
        header.setStretchLastSection(False)    # 末列也可拖动（可选）
        self.model = PandasModel()
        self.table.setModel(self.model)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)

        splitter.addWidget(self.table)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    # ---------- 菜单 ----------
    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("文件")
        m_file.addAction("打开 Excel", self.open_excel)
        m_file.addAction("导出结果", self.export_excel)

        m_tool = mb.addMenu("工具")
        m_tool.addAction("打开日志目录", lambda: open_path(LOG_DIR))
        m_tool.addAction("清理 30 天前缓存", lambda: self.cache.cleanup(30))

    # ---------- 文件 ----------
    def open_excel(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 Excel 文件", str(RUN_DIR), "Excel (*.xlsx *.xls)"
        )
        if paths:
            self.load_excels(paths)

    def load_excels(self, paths: List[str]):
        try:
            self.df = pd.concat([pd.read_excel(p) for p in paths], ignore_index=True)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self.model.setDataFrame(self.df)
        self.list_cols.clear()
        for c in self.df.columns:
            self.list_cols.addItem(c)

    def export_excel(self):
        if self.df.empty:
            QMessageBox.information(self, "提示", "暂无数据可导出")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel",
            f"result_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            "Excel (*.xlsx)"
        )
        if path:
            try:
                self.df.to_excel(path, index=False)
                QMessageBox.information(self, "成功", f"已导出到 {path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    # ---------- 解析 ----------
    def parse_selected(self):
        if self.df.empty:
            QMessageBox.information(self, "提示", "请先打开 Excel 文件")
            return
        cols = [i.text() for i in self.list_cols.selectedItems()]
        if not cols:
            QMessageBox.information(self, "提示", "请至少选择一列")
            return

        ips_series = pd.concat([self.df[c].astype(str) for c in cols])
        ips = (ips_series.str.strip()
               .replace("", pd.NA)
               .dropna()
               .unique()
               .tolist())

        cached = self.cache.fetch(ips)
        missing = [ip for ip in ips if ip not in cached]
        log.info("总 %d IP，缓存命中 %d，待解析 %d", len(ips), len(cached), len(missing))

        if not missing:
            self._merge_result(cached)
            return

        self.pbar.setMaximum(len(missing))
        self.pbar.setValue(0)
        self.pbar.show()

        self.thread = QThread()
        self.worker = ParseWorker(missing)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _on_finished(self, data: Dict[str, Dict]):
        self.pbar.hide()
        self.cache.save(data)
        self._merge_result(data)

    def _merge_result(self, new_data: Dict[str, Dict]):
        join = lambda d: " ".join([d.get("country", ""), d.get("region", ""), d.get("city", "")]).strip()
        mapping = {ip: join(v) for ip, v in new_data.items()}

        for it in self.list_cols.selectedItems():
            col = it.text()
            self.df[f"{col}_归属地"] = self.df[col].astype(str).map(mapping).fillna("")

        self.model.setDataFrame(self.df)

    # ---------- 表格右键 ----------
    def _table_menu(self, pos):
        idx = self.table.indexAt(pos)
        if not idx.isValid():
            return
        col = idx.column()
        col_name = self.df.columns[col]

        menu = QMenu(self)
        menu.addAction("复制单元格",
                       lambda: QApplication.clipboard().setText(idx.data()))

        def copy_col():
            QApplication.clipboard().setText(
                "\n".join(self.df.iloc[:, col].astype(str).tolist())
            )
        menu.addAction(f"复制整列 [{col_name}]", copy_col)
        menu.exec_(self.table.viewport().mapToGlobal(pos))

    # ---------- 拖拽 ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith((".xlsx", ".xls"))]
        if paths:
            self.load_excels(paths)

# ─────────────── utils ───────────────
def open_path(p: Path):
    import subprocess, os, platform
    if platform.system() == "Windows":
        os.startfile(str(p))
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])

def excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    log.error(msg)
    QMessageBox.critical(None, "未捕获异常", msg)
    sys.exit(1)

sys.excepthook = excepthook

# ─────────────── main ───────────────
def main():
    app = QApplication(sys.argv)
    if app.palette().color(QPalette.Window).value() < 128:
        app.setStyleSheet("QMainWindow{background:#2b2b2b;color:#ddd;}")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()