#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IP 归属地批量查询工具 – 进阶版
功能：
    • 亮 / 暗 / 跟随系统主题热切换
    • 最近文件、窗口状态记忆
    • 多文件拖拽 / 批量选择，自动合并
    • 表格右键复制单元格 / 整列
    • SQLite 本地缓存 + 并发请求 + 进度条
    • 日志系统、缓存清理、配置热加载
"""

from __future__ import annotations
import sys, os, time, math, json, logging, logging.handlers, configparser, sqlite3
import threading, concurrent.futures, shutil, webbrowser
from pathlib import Path
from itertools import chain
from urllib.parse import quote
from typing import List, Dict

import pandas as pd
import requests

from PyQt5.QtCore import (
    Qt, QSettings, QFileSystemWatcher, QTimer, QSize,
    QAbstractTableModel, QModelIndex, QObject, QThread, pyqtSignal
)
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QAction, QFileDialog, QSplitter,
    QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QTableView,
    QHeaderView, QStyle, QPushButton, QProgressBar
)

# ===============================================================
# 0. 基础常量 & 目录
# ===============================================================
APP_NAME    = "IPLocator"
ORG_NAME    = "DemoCorp"
DEFAULT_APP_ID = "ipsearch"
API_URL        = "https://www.hyhdt.com/api/getipaddress.ashx"

BASE_DIR  = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
CONFIG_FILE = BASE_DIR / "config.ini"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ===============================================================
# 1. 日志系统
# ===============================================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=2*1024*1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)

    logger.addHandler(fh); logger.addHandler(sh)
    return logger

log = setup_logger()
log.info("程序启动")

# ===============================================================
# 2. 配置文件：生成 / 热加载
# ===============================================================
class ConfigManager:
    def __init__(self):
        self._parser = configparser.RawConfigParser()
        self._watcher = QFileSystemWatcher()
        self._watcher.fileChanged.connect(self._on_changed)

        if not CONFIG_FILE.exists():
            self._create_default()
            QMessageBox.information(
                None, "首次运行",
                f"已生成 {CONFIG_FILE.name}\n请填入 appkey 后重新启动。")
            sys.exit(0)

        self._load()
        self._watcher.addPath(str(CONFIG_FILE))

    @property
    def appid(self):  return self._parser["credential"]["appid"]
    @property
    def appkey(self): return self._parser["credential"]["appkey"]

    def _create_default(self):
        self._parser["credential"] = {"appid":DEFAULT_APP_ID, "appkey":"请在此处填写你的 appkey"}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            self._parser.write(f)

    def _load(self):
        self._parser.read(CONFIG_FILE, encoding="utf-8")
        if "credential" not in self._parser or \
           self._parser["credential"].get("appkey","").strip() in ("","请在此处填写你的 appkey"):
            QMessageBox.warning(None,"AppKey 未设置",f"{CONFIG_FILE.name} 中 appkey 为空")
            sys.exit(0)
        log.info("配置文件已加载")

    def _on_changed(self, _):
        log.info("检测到 config.ini 修改，重新加载")
        QTimer.singleShot(300, self._load)

# ===============================================================
# 3. 主题系统
# ===============================================================
class Theme:
    LIGHT="light"; DARK="dark"; FOLLOW="follow"

_LIGHT_QSS = """
QMainWindow         { background:#f8f9fa; color:#212529; }
QMenuBar            { background:#e9ecef; }
QMenuBar::item      { padding:4px 8px; spacing:4px; }
QMenuBar::item:selected { background:#ced4da; }
QMenu               { background:white; }
QMenu::item:selected{ background:#3391ff; color:white; }

QPushButton{
    background:#007bff; color:white; border:none; border-radius:4px; padding:6px 12px;
}
QPushButton:hover   { background:#3391ff; }
QPushButton:disabled{ background:#9ec5ff; }

QTableView          { gridline-color:#dee2e6; selection-background-color:#cce5ff; }
QHeaderView::section{
    background:#e9ecef; padding:4px; border:none; border-right:1px solid #dee2e6; font-weight:bold;
}
QScrollBar:vertical{ background:#f1f3f5; width:12px; margin:0; }
QScrollBar::handle:vertical{ background:#adb5bd; min-height:20px; border-radius:6px; }
QProgressBar{ border:none; background:#e9ecef; height:16px; border-radius:8px; }
QProgressBar::chunk{ background:#28a745; border-radius:8px; }
"""

_DARK_QSS = """
QMainWindow         { background:#1e1e1e; color:#e0e0e0; }
QMenuBar            { background:#2d2d30; color:#e0e0e0; }
QMenuBar::item:selected { background:#3e3e42; }
QMenu               { background:#2d2d30; color:#e0e0e0; }
QMenu::item:selected{ background:#0a84ff; color:white; }

QPushButton{
    background:#0a84ff; color:white; border:none; border-radius:4px; padding:6px 12px;
}
QPushButton:hover   { background:#3b9cff; }
QPushButton:disabled{ background:#556677; }

QTableView          { gridline-color:#3e3e42; selection-background-color:#094771; }
QHeaderView::section{
    background:#2d2d30; padding:4px; border:none; border-right:1px solid #3e3e42; font-weight:bold;
}
QScrollBar:vertical{ background:#2d2d30; width:12px; margin:0; }
QScrollBar::handle:vertical{ background:#55595f; min-height:20px; border-radius:6px; }
QProgressBar{ border:none; background:#3e3e42; height:16px; border-radius:8px; }
QProgressBar::chunk{ background:#20c997; border-radius:8px; }
"""

class ThemeManager:
    def __init__(self, app: QApplication, settings: QSettings):
        self.app, self.settings = app, settings
        self.current = settings.value("ui/theme", Theme.FOLLOW, str)
        self.apply(self.current)

    def apply(self, mode: str):
        if mode==Theme.LIGHT:
            self.app.setStyleSheet(_LIGHT_QSS)
        elif mode==Theme.DARK:
            self.app.setStyleSheet(_DARK_QSS)
        else:  # 跟随系统
            palette = self.app.palette()
            self.app.setStyleSheet(_DARK_QSS if palette.color(QPalette.Window).lightness()<128 else _LIGHT_QSS)
        self.current = mode
        self.settings.setValue("ui/theme", mode)
        log.info("主题切换 -> %s", mode)

# ===============================================================
# 4. Pandas → Qt Model
# ===============================================================
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame = pd.DataFrame(), parent=None):
        super().__init__(parent); self._df = df
    def rowCount(self,_=QModelIndex()):    return len(self._df.index)
    def columnCount(self,_=QModelIndex()): return len(self._df.columns)
    def data(self, idx, role=Qt.DisplayRole):
        if idx.isValid() and role==Qt.DisplayRole:
            return str(self._df.iat[idx.row(), idx.column()])
    def headerData(self, sec, orient, role=Qt.DisplayRole):
        if role!=Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if orient==Qt.Horizontal else str(self._df.index[sec])
    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel(); self._df = df; self.endResetModel()

# ===============================================================
# 5. 缓存层 (SQLite)
# ===============================================================
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

    def fetch(self, ips: list[str]) -> dict[str, dict]:
        if not ips:
            return {}
        q = "SELECT ip,country,region,city FROM ip_cache WHERE ip IN (%s)" % (
            ",".join("?" * len(ips))
        )
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(q, ips)
            return {
                ip: dict(country=c, region=r, city=ci)
                for ip, c, r, ci in cur.fetchall()
            }

    def save(self, mapping: dict[str, dict]):
        if not mapping:
            return
        rows = [
            (
                ip,
                d.get("country", ""),
                d.get("region", ""),
                d.get("city", ""),
                int(time.time()),
            )
            for ip, d in mapping.items()
        ]
        with self.lock:
            self.conn.executemany("REPLACE INTO ip_cache VALUES(?,?,?,?,?)", rows)
            self.conn.commit()

    def cleanup(self, days: int = 30):
        limit = int(time.time() - days * 86400)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM ip_cache WHERE ts<?", (limit,))
            self.conn.commit()
            log.info("缓存清理删除 %d 条", cur.rowcount)

# ===============================================================
# 6. 网络 & Worker
# ===============================================================
def build_url(appid:str, appkey:str, ips:list[str])->str:
    enc=lambda ip: ip if ":" not in ip else quote(ip,safe="")
    return f"{API_URL}?appid={appid}&appkey={appkey}&ips={','.join(enc(i) for i in ips)}"

def query_ips(appid:str, appkey:str, ips:list[str])->dict[str,dict]:
    r=requests.get(build_url(appid,appkey,ips), timeout=10)
    r.raise_for_status(); j=r.json()
    if j.get("code")!=200: raise RuntimeError(j.get("msg","接口错误"))
    return {d["ip"]:d for d in j["data"]}

class ParseWorker(QObject):
    progress=pyqtSignal(int,int)
    finished=pyqtSignal(dict)
    failed  =pyqtSignal(str)
    def __init__(self, appid, appkey, ips_missing, cached, cache_db, max_workers=4):
        super().__init__()
        self.appid, self.appkey = appid, appkey
        self.missing=ips_missing; self.cached=cached
        self.cache_db=cache_db; self.max_workers=max_workers
    def run(self):
        try:
            if not self.missing:
                self.finished.emit(self.cached); return
            chunks=[self.missing[i:i+50] for i in range(0,len(self.missing),50)]
            total=len(chunks); done=0; lock=threading.Lock()
            new_res={}
            def task(chunk):
                data=query_ips(self.appid,self.appkey,chunk)
                nonlocal done
                with lock:
                    done+=1; self.progress.emit(done,total)
                return data
            with concurrent.futures.ThreadPoolExecutor(self.max_workers) as pool:
                for data in pool.map(task, chunks):
                    new_res.update(data)
            if new_res: self.cache_db.save(new_res)
            self.finished.emit(self.cached | new_res)
        except Exception as e:
            self.failed.emit(str(e))

# ===============================================================
# 7. 主窗口
# ===============================================================
class MainWindow(QMainWindow):
    def __init__(self, cfg:ConfigManager, settings:QSettings, theme_mgr:ThemeManager):
        super().__init__()
        self.cfg, self.settings, self.theme_mgr = cfg, settings, theme_mgr
        self.cache=CacheDB(BASE_DIR/"cache.db")
        self.recent_max=5

        # Data
        self.df=pd.DataFrame(); self.model=PandasModel()

        # ---------------- 界面 ----------------
        self.setWindowTitle("IP 归属地批量查询工具"); self.resize(1100,650)
        self.setAcceptDrops(True)

        # 菜单
        mbar=self.menuBar()
        m_file=mbar.addMenu("文件(&F)")
        m_theme=mbar.addMenu("主题(&M)")

        act_open=QAction("打开(&O)…",self,triggered=self._open_dialog)
        act_export=QAction("导出(&E)…",self,triggered=self._export,enabled=False)
        act_exit=QAction("退出(&Q)",self,triggered=self.close)
        m_file.addAction(act_open)
        self._sep_recent=m_file.addSeparator(); self._recent_acts=[]
        m_file.addSeparator(); m_file.addAction(act_export)
        m_file.addSeparator(); m_file.addAction(act_exit)
        self.act_export=act_export

        for text,mode in [("亮色",Theme.LIGHT),("暗色",Theme.DARK),("跟随系统",Theme.FOLLOW)]:
            act=QAction(text,self,checkable=True,
                        triggered=lambda _,m=mode:self.theme_mgr.apply(m))
            m_theme.addAction(act); act.setChecked(mode==self.theme_mgr.current)

        # 中央区
        splitter=QSplitter(self); self.setCentralWidget(splitter)
        # 左
        left=QWidget(); splitter.addWidget(left)
        lv=QVBoxLayout(left); lv.setContentsMargins(10,10,10,10); lv.setSpacing(12)

        self.lab_file=QLabel("未选择文件"); self.lab_file.setStyleSheet("color:#6c757d;"); self.lab_file.setWordWrap(True)
        lv.addWidget(self.lab_file)

        self.list_cols=QListWidget(); self.list_cols.setSelectionMode(QListWidget.MultiSelection)
        lv.addWidget(self.list_cols,1)

        self.btn_parse=QPushButton(" 解析 IP 归属",icon=QApplication.style().standardIcon(QStyle.SP_ArrowRight))
        self.btn_parse.setEnabled(False); self.btn_parse.clicked.connect(self._parse)
        lv.addWidget(self.btn_parse)

        self.progress=QProgressBar(); self.progress.setVisible(False)
        lv.addWidget(self.progress); lv.addStretch()

        # 右
        self.table=QTableView(); splitter.addWidget(self.table)
        self.table.setModel(self.model); self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)

        # 恢复窗口/最近文件
        self._restore_state()
        # 工具菜单
        self._init_tools_menu()

    # ---------- 工具菜单 ----------
    def _init_tools_menu(self):
        m_tools=self.menuBar().addMenu("工具(&T)")
        act_log=QAction("打开日志目录",self,triggered=lambda: webbrowser.open(str(LOG_DIR.resolve())))
        act_clean=QAction("清理30天前缓存",self,
            triggered=lambda: ( self.cache.cleanup(30), QMessageBox.information(self,"完成","缓存清理完成") ))
        m_tools.addAction(act_log); m_tools.addAction(act_clean)

    # ---------- 拖拽 ----------
    def dragEnterEvent(self,e):
        if e.mimeData().hasUrls() and all(u.toLocalFile().lower().endswith(".xlsx") for u in e.mimeData().urls()):
            e.acceptProposedAction()
    def dropEvent(self,e):
        self._load_files([u.toLocalFile() for u in e.mimeData().urls()])

    # ---------- 文件 ----------
    def _open_dialog(self):
        paths,_=QFileDialog.getOpenFileNames(self,"选择 Excel 文件","","Excel (*.xlsx)")
        if paths: self._load_files(paths)

    def _load_files(self,paths:List[str]):
        dfs=[]; succeeded=[]
        for p in paths:
            try:
                df=pd.read_excel(p,dtype=str); df["__源文件"]=os.path.basename(p)
                dfs.append(df); succeeded.append(p); log.info("读取 %d 行 %s",len(df),p)
            except Exception as e:
                QMessageBox.warning(self,"读取失败",f"{p}\n{e}")
        if not dfs: return
        self.df=pd.concat(dfs,ignore_index=True); self.model.setDataFrame(self.df)
        self.lab_file.setText(" , ".join(os.path.basename(p) for p in succeeded))
        self._populate_cols()
        self.btn_parse.setEnabled(True); self.act_export.setEnabled(False)
        for p in succeeded: self._add_recent(p)

    def _populate_cols(self):
        self.list_cols.clear()
        for c in self.df.columns:
            if c!="__源文件": self.list_cols.addItem(QListWidgetItem(c))

    # ---------- 表格右键 ----------
    # --- 替换 MainWindow 中的 _table_menu 方法 ---
    def _table_menu(self, pos):
        idx = self.table.indexAt(pos)
        if not idx.isValid():
            return

        menu = self.table.viewport().createStandardContextMenu()
        menu.clear()

        # 复制单元格
        menu.addAction(
            "复制单元格",
            lambda: QApplication.clipboard().setText(idx.data())
        )

        # 复制整列
        col_name = self.df.columns[idx.column()]
        menu.addAction(
            f"复制整列 [{col_name}]",
            lambda: QApplication.clipboard().setText(
                "\n".join(self.df.iloc[:, idx.column()].astype(str).tolist())
            )
        )

        menu.exec_(self.table.viewport().mapToGlobal(pos))

    # ---------- 解析 ----------
    def _parse(self):
        cols=[it.text() for it in self.list_cols.selectedItems()]
        if not cols:
            QMessageBox.information(self,"提示","请勾选需要解析的列"); return
        uniq=list({ip for ip in chain.from_iterable(self.df[c].dropna().astype(str) for c in cols)})
        if not uniq:
            QMessageBox.information(self,"提示","所选列无 IP"); return
        cached=self.cache.fetch(uniq); missing=[ip for ip in uniq if ip not in cached]
        log.info("需解析 %d，已缓存 %d",len(missing),len(cached))

        # UI
        self.btn_parse.setEnabled(False); self.progress.setVisible(True)
        self.progress.setValue(0); self.progress.setMaximum(max(math.ceil(len(missing)/50),1))

        # 线程
        self.th=QThread(self)
        self.worker=ParseWorker(self.cfg.appid,self.cfg.appkey,missing,cached,self.cache)
        self.worker.moveToThread(self.th)
        self.th.started.connect(self.worker.run)
        self.worker.progress.connect(lambda d,t: self.progress.setValue(d))
        self.worker.finished.connect(lambda res: self._on_parse_done(res,cols))
        self.worker.failed.connect(self._on_parse_fail)
        self.worker.finished.connect(self.th.quit); self.worker.failed.connect(self.th.quit)
        self.th.finished.connect(self.th.deleteLater)
        self.th.start()

    def _on_parse_done(self,res:dict,cols:List[str]):
        for c in cols:
            self.df[f"{c}_归属"]=self.df[c].map(
                lambda ip: f"{res.get(str(ip),{}).get('country','')}"
                           f"{res.get(str(ip),{}).get('region','')}"
                           f"{res.get(str(ip),{}).get('city','')}")
        self.model.setDataFrame(self.df)
        self.progress.setVisible(False); self.btn_parse.setEnabled(True); self.act_export.setEnabled(True)
        QMessageBox.information(self,"完成","IP 归属解析完成")

    def _on_parse_fail(self,msg):
        log.error("解析失败 %s",msg)
        self.progress.setVisible(False); self.btn_parse.setEnabled(True)
        QMessageBox.critical(self,"解析失败",msg)

    # ---------- 导出 ----------
    def _export(self):
        f,_=QFileDialog.getSaveFileName(self,"导出 Excel","ip_result.xlsx","Excel (*.xlsx)")
        if not f: return
        if not f.lower().endswith(".xlsx"): f+=".xlsx"
        try: self.df.to_excel(f,index=False); QMessageBox.information(self,"导出成功",f"已导出到\n{f}")
        except Exception as e: QMessageBox.critical(self,"导出失败",str(e))

    # ---------- 最近文件 ----------
    def _add_recent(self,path:str):
        files=self.settings.value("recent",[],list)
        if path in files: files.remove(path)
        files.insert(0,path); self.settings.setValue("recent",files[:self.recent_max])
        self._refresh_recent()

    def _refresh_recent(self):
        files=self.settings.value("recent",[],list)
        m=self.menuBar().actions()[0].menu()
        for act in self._recent_acts: m.removeAction(act)
        self._recent_acts.clear()
        for i,p in enumerate(files):
            act=QAction(f"{i+1}. {Path(p).name}",self,triggered=lambda _,pp=p:self._load_files([pp]))
            m.insertAction(self._sep_recent,act); self._recent_acts.append(act)

    # ---------- 状态 ----------
    def _restore_state(self):
        self.restoreGeometry(self.settings.value("ui/geo",b""))
        self.restoreState(self.settings.value("ui/state",b""))
        self._refresh_recent()

    def closeEvent(self,e):
        self.settings.setValue("ui/geo",self.saveGeometry())
        self.settings.setValue("ui/state",self.saveState())
        super().closeEvent(e)

# ===============================================================
# 8. 小提示一次性
# ===============================================================
def show_tip_once():
    flag=BASE_DIR/".tip_seen"
    if flag.exists(): return
    QMessageBox.information(None,"小提示","可在菜单“主题”中随时切换亮 / 暗 / 跟随系统主题。")
    flag.touch()

# ===============================================================
# 9. 入口
# ===============================================================
def main():
    app=QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME); app.setApplicationName(APP_NAME)

    settings=QSettings(ORG_NAME,APP_NAME)
    cfg=ConfigManager(); theme_mgr=ThemeManager(app,settings)

    win=MainWindow(cfg,settings,theme_mgr); win.show()
    show_tip_once()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()