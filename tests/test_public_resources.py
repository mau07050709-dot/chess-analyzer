"""Public resources work without access to private screenshots or source."""
import unittest
from pathlib import Path
import numpy as np
from ajedrez.paths import resource_root
from ajedrez.recognition import _load_classifier
from ajedrez import ffa_engine

class PublicResourceTests(unittest.TestCase):
    def test_native_ffa_module_is_loaded_from_this_export(self):
        root=resource_root()
        for module in ffa_engine.modules():
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(root))
        result=ffa_engine.analyze(ffa_engine.new_position(),100)
        self.assertEqual(result.player,'red')

    def test_local_onnx_model_runs_and_session_is_reused(self):
        path=str(resource_root()/'models/chess-tiles-v2.onnx')
        session=_load_classifier(path)
        result=session.run(None,{session.get_inputs()[0].name:np.zeros((64,1024),np.float32)})[0]
        self.assertEqual(result.shape,(64,13));self.assertTrue(np.isfinite(result).all())
        self.assertIs(session,_load_classifier(path))

    def test_imported_application_source_stays_in_export(self):
        import ajedrez,ajedrez.ui.main_window
        self.assertTrue(Path(ajedrez.__file__).resolve().is_relative_to(resource_root()))
        self.assertTrue(Path(ajedrez.ui.main_window.__file__).resolve().is_relative_to(resource_root()))
