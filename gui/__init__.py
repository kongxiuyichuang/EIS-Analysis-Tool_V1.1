import sys, os

def resource_path(relative_path: str) -> str:
    """返回资源文件的绝对路径，兼容开发环境与 PyInstaller 打包"""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)
