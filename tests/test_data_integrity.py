import unittest,tempfile
from pathlib import Path
from src.data import Case,sha256_file

class DataIntegrityTests(unittest.TestCase):
    def test_changed_annotation_bytes_are_rejected(self):
        from src.data import verify_case_files
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'file.png';path.write_bytes(b'original')
            case=Case('a','test','validation',str(path),str(path),str(path),
                      {'image':sha256_file(path),'target':sha256_file(path),'valid':sha256_file(path)})
            verify_case_files(case)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_case_files(case)

if __name__=='__main__':unittest.main()
