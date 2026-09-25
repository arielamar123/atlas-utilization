import logging
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.executor import PipelineExecutor


class InvariantMassStatsTests(unittest.TestCase):
    signature_a = "parsed_FS_1e_0m_0j_0g_0t_0b_IM_e0e1"
    signature_b = "parsed_FS_1e_0m_0j_0g_0t_0b_IM_e0j0"
    final_state = "1e_0m_0j_0g_0t_0b"

    def _executor(self):
        executor = PipelineExecutor.__new__(PipelineExecutor)
        executor.logger = logging.getLogger("test-im-array-stats")
        return executor

    def _shard(self, directory, name, counts=None, entries=(100, 100)):
        path = Path(directory, name)
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE array_chunks(signature TEXT, n_entries INTEGER)"
            )
            connection.executemany(
                "INSERT INTO array_chunks VALUES (?, ?)",
                [(self.signature_a, entries[0]), (self.signature_b, entries[1])],
            )
            if counts is not None:
                connection.execute(
                    "CREATE TABLE final_state_counts(final_state TEXT, n_events INTEGER)"
                )
                connection.execute(
                    "INSERT INTO final_state_counts VALUES (?, ?)",
                    (self.final_state, counts),
                )
            connection.commit()
        finally:
            connection.close()
        return path

    def test_exact_sqlite_counts_are_not_multiplied_by_im_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            self._shard(directory, "one.sqlite", counts=100)
            stats = self._executor()._read_im_array_stats(directory)
        self.assertEqual(stats["events_per_final_state"][self.final_state], 100)
        self.assertEqual(stats["total_events_processed"], 100)
        self.assertEqual(stats["total_mass_values"], 200)

    def test_exact_counts_sum_across_shards_independently_of_mass_values(self):
        with tempfile.TemporaryDirectory() as directory:
            self._shard(directory, "one.sqlite", counts=100, entries=(100, 100))
            self._shard(directory, "two.sqlite", counts=25, entries=(25, 25))
            stats = self._executor()._read_im_array_stats(directory)
        self.assertEqual(stats["events_per_final_state"][self.final_state], 125)
        self.assertEqual(stats["total_events_processed"], 125)
        self.assertEqual(stats["total_mass_values"], 250)

    def test_legacy_sqlite_uses_non_multiplicative_final_state_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self._shard(directory, "legacy.sqlite", counts=None, entries=(100, 100))
            stats = self._executor()._read_im_array_stats(directory)
        self.assertEqual(stats["events_per_final_state"][self.final_state], 100)
        self.assertEqual(stats["total_events_processed"], 100)
        self.assertEqual(stats["total_mass_values"], 200)


if __name__ == "__main__":
    unittest.main()
