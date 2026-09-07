import unittest

from domain.config import PipelineConfig
from orchestration.handlers.parsing_handler import randomize_metadata_file_order
from utils.batching import get_batch_slice_by_year


class RandomizedFileOrderTests(unittest.TestCase):
    def test_seeded_order_is_reproducible_without_mutating_metadata(self):
        metadata = {
            "2024r-pp": [f"file_{index}" for index in range(20)],
            "2020e-13tev": [f"old_file_{index}" for index in range(10)],
        }
        original = {key: list(urls) for key, urls in metadata.items()}

        first = randomize_metadata_file_order(metadata, seed=42)
        second = randomize_metadata_file_order(metadata, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(metadata, original)
        self.assertNotEqual(first["2024r-pp"], metadata["2024r-pp"])
        for release_year in metadata:
            self.assertCountEqual(first[release_year], metadata[release_year])

    def test_shuffle_before_batching_keeps_batch_assignments_disjoint(self):
        metadata = {"2024r-pp": [f"file_{index}" for index in range(21)]}
        randomized = randomize_metadata_file_order(metadata, seed=7)

        first_batch = get_batch_slice_by_year(randomized, 1, 2)["2024r-pp"]
        second_batch = get_batch_slice_by_year(randomized, 2, 2)["2024r-pp"]

        self.assertFalse(set(first_batch) & set(second_batch))
        self.assertCountEqual(first_batch + second_batch, metadata["2024r-pp"])

    def test_config_loads_randomization_options(self):
        config = PipelineConfig.from_dict(
            {
                "tasks": {"do_parsing": True},
                "parsing_task_config": {
                    "output_path": "parsed",
                    "file_urls_path": "metadata.json",
                    "jobs_logs_path": "logs",
                    "randomize_file_order": True,
                    "file_order_random_seed": 123,
                },
            }
        )

        self.assertTrue(config.parsing_config.randomize_file_order)
        self.assertEqual(config.parsing_config.file_order_random_seed, 123)


if __name__ == "__main__":
    unittest.main()
