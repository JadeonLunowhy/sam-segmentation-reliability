import unittest,tempfile
from pathlib import Path
from scripts.build_materials import build_pdf,p

class MaterialTests(unittest.TestCase):
    def test_pdf_export_preserves_editable_source_story(self):
        story=[p('Editable research paragraph.')]
        with tempfile.TemporaryDirectory() as folder:
            build_pdf(Path(folder)/'test.pdf',story)
        self.assertEqual(len(story),1)
        self.assertEqual(story[0].getPlainText(),'Editable research paragraph.')

if __name__=='__main__':unittest.main()
