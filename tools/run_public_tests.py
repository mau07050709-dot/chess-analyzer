"""Run only public-source tests; report unavailable private-fixture modules."""
import ast
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.chdir(ROOT)

def fixture_dependencies():
    files={f'tests.{f.stem}':f for f in (ROOT/'tests').glob('*.py')}
    blocked=set();dependencies={}
    for name,path in files.items():
        text=path.read_text(encoding='utf-8-sig');tree=ast.parse(text)
        if 'fixtures' in text:blocked.add(name)
        dependencies[name]={node.module for node in ast.walk(tree)
                            if isinstance(node,ast.ImportFrom) and node.module in files}
    while True:
        expanded=blocked|{name for name,deps in dependencies.items() if deps&blocked}
        if expanded==blocked:return blocked
        blocked=expanded

def leaves(suite):
    for case in suite:
        if isinstance(case,unittest.TestSuite):yield from leaves(case)
        else:yield case

if __name__=='__main__':
    blocked=fixture_dependencies()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
    selected=unittest.TestSuite()
    for case in leaves(suite):
        if type(case).__module__ in blocked:
            def unavailable():raise unittest.SkipTest('Private screenshot fixture dependency intentionally excluded')
            replacement=unittest.FunctionTestCase(unavailable,description=case.id())
            selected.addTest(replacement)
        else:selected.addTest(case)
    with patch('ajedrez.ui.main_window.left_mouse_button_down',return_value=False),patch(
            'ajedrez.ui.four_player_tab.left_mouse_button_down',return_value=False):
        result=unittest.TextTestRunner(verbosity=2).run(selected)
    raise SystemExit(not result.wasSuccessful())
