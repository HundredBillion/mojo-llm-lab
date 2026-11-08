import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from summarize_cuda_profile import summarize


class ProfileSummaryTests(unittest.TestCase):
    def event(self, **changes):
        return dict(dict(kernel='matvec', start_ns=0, end_ns=10, grid=[1,1,1], block=[32,4,1]), **changes)

    def test_groups_dimensions_and_preserves_overlapping_kernel_time(self):
        result = summarize([self.event(), self.event(start_ns=5,end_ns=15), self.event(grid=[2,1,1],end_ns=20)])
        self.assertEqual(result['kernel_calls'],3)
        self.assertEqual(result['total_gpu_ns'],40)
        self.assertEqual([r['calls'] for r in result['groups']],[2,1])
        self.assertEqual([r['share_percent'] for r in result['groups']],[50,50])

    def test_missing_names_bad_times_and_bad_dimensions_fail(self):
        for change in ({'kernel':None},{'end_ns':0},{'end_ns':-1},{'start_ns':True},{'grid':[1,1]}, {'block':[32,0,1]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                summarize([self.event(**change)])
        with self.assertRaises(ValueError):summarize([])

    def test_sqlite_export_does_not_silently_drop_unknown_kernel_names(self):
        from contextlib import closing
        import sqlite3
        import tempfile
        from summarize_cuda_profile import sqlite_events
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.sqlite'
            with closing(sqlite3.connect(path)) as c:
                c.execute('CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT)')
                c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(demangledName INTEGER,start INTEGER,end INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER)')
                c.execute("INSERT INTO StringIds VALUES(1,'known')")
                c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,0,10,1,1,1,32,1,1)',[(1,),(2,)])
                c.commit()
            events = list(sqlite_events(path))
            self.assertEqual(len(events),2)
            with self.assertRaises(ValueError):summarize(events)
