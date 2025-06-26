#!/usr/bin/env python3
"""
Data processing infrastructure for ATLAS dijet analysis.
Handles large datasets with streaming and chunked processing.
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import h5py
import pandas as pd
import psutil
import uproot
import zarr
from dask.delayed import delayed
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""

    chunk_size: int = 50000  # Events per chunk
    max_memory_gb: float = 8.0  # Max memory usage
    n_workers: int = 4  # Number of workers
    cache_dir: str = "cache"  # Cache directory
    output_format: str = "parquet"  # Output format
    compression: str = "snappy"  # Compression
    progress_interval: int = 10000  # Progress update interval


class MemoryMonitor:
    """Monitor memory usage."""

    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.peak_memory = 0

    def check_memory(self) -> Dict[str, float]:
        """Return current memory usage."""
        process = psutil.Process()
        current_memory = process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)

        return {
            "current_gb": current_memory / 1024**3,
            "peak_gb": self.peak_memory / 1024**3,
            "available_gb": (psutil.virtual_memory().available) / 1024**3,
            "usage_fraction": current_memory / self.max_memory_bytes,
        }

    def is_memory_safe(self) -> bool:
        """Return True if memory usage is within safe limits."""
        return self.check_memory()["usage_fraction"] < 0.8


class LargeScaleATLASProcessor:
    """
    Memory-efficient processor for large ATLAS datasets.

    Handles datasets that don't fit in memory using streaming,
    chunking, and distributed processing.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def estimate_dataset_size(self, file_paths: List[str]) -> Dict[str, float]:
        """Estimate total dataset size and memory requirements."""
        total_size_bytes = sum(
            Path(f).stat().st_size for f in file_paths if Path(f).exists()
        )
        total_size_gb = total_size_bytes / 1024**3

        estimated_memory_gb = total_size_gb * 3
        n_chunks_needed = max(1, int(estimated_memory_gb / self.config.max_memory_gb))

        return {
            "file_size_gb": total_size_gb,
            "estimated_memory_gb": estimated_memory_gb,
            "n_chunks_recommended": n_chunks_needed,
            "chunk_size_recommended": max(
                1000, self.config.chunk_size // n_chunks_needed
            ),
        }

    def stream_atlas_files(
        self, file_paths: List[str], branches: List[str]
    ) -> Iterator[pd.DataFrame]:
        """
        Stream ATLAS files in chunks.
        Args:
            file_paths: List of ROOT file paths
            branches: List of branches to read
        Yields:
            DataFrame chunks
        """
        total_files = len(file_paths)
        events_processed = 0

        with tqdm(
            total=total_files, desc="Processing Files", unit="files"
        ) as file_pbar:
            for file_idx, file_path in enumerate(file_paths):
                try:
                    if not self.memory_monitor.is_memory_safe():
                        logger.warning("Memory usage high, forcing garbage collection")
                        gc.collect()

                    root_file = uproot.open(file_path)
                    tree = root_file["CollectionTree"]
                    total_entries = len(tree)

                    for start_idx in range(0, total_entries, self.config.chunk_size):
                        end_idx = min(start_idx + self.config.chunk_size, total_entries)

                        chunk_data = tree.arrays(
                            branches,
                            entry_start=start_idx,
                            entry_stop=end_idx,
                            library="pd",
                        )

                        chunk_data["file_idx"] = file_idx
                        chunk_data["chunk_start"] = start_idx
                        events_processed += len(chunk_data)

                        memory_info = self.memory_monitor.check_memory()
                        if memory_info["usage_fraction"] > 0.9:
                            logger.error(
                                "Memory usage critical! Consider reducing chunk size."
                            )
                            raise MemoryError(
                                f"Memory usage: {memory_info['current_gb']:.1f}GB"
                            )

                        yield chunk_data

                        if events_processed % self.config.progress_interval == 0:
                            logger.info(
                                f"Processed {events_processed:,} events, "
                                f"Memory: {memory_info['current_gb']:.1f}GB"
                            )

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

                file_pbar.update(1)
                file_pbar.set_postfix({"events": f"{events_processed:,}"})

    def cache_processed_chunks(
        self, file_paths: List[str], processor_func: Callable
    ) -> List[str]:
        """
        Cache processed chunks to disk for faster reprocessing.

        Args:
            file_paths: List of input files
            processor_func: Function to process each chunk

        Returns:
            List of cached chunk file paths
        """
        cache_files = []

        branches = [
            "AnalysisJetsAuxDyn.pt",
            "AnalysisJetsAuxDyn.eta",
            "AnalysisJetsAuxDyn.phi",
            "AnalysisJetsAuxDyn.m",
            "AnalysisJetsAuxDyn.DFCommonJets_fJvt",
        ]

        chunk_idx = 0
        for chunk_df in self.stream_atlas_files(file_paths, branches):
            processed_chunk = processor_func(chunk_df)

            cache_file = (
                self.cache_dir
                / f"processed_chunk_{chunk_idx:06d}.{self.config.output_format}"
            )

            if self.config.output_format == "parquet":
                processed_chunk.to_parquet(
                    cache_file, compression=self.config.compression
                )
            elif self.config.output_format == "hdf5":
                processed_chunk.to_hdf(cache_file, key="data", mode="w", complevel=9)
            elif self.config.output_format == "feather":
                processed_chunk.to_feather(cache_file)

            cache_files.append(str(cache_file))
            chunk_idx += 1

            logger.info(f"Cached chunk {chunk_idx} to {cache_file}")

        return cache_files

    def load_cached_chunks(self, cache_files: List[str]) -> Iterator[pd.DataFrame]:
        """Load cached chunks for further processing."""
        for cache_file in tqdm(cache_files, desc="Loading Cached Chunks"):
            if self.config.output_format == "parquet":
                yield pd.read_parquet(cache_file)
            elif self.config.output_format == "hdf5":
                df = pd.DataFrame(pd.read_hdf(cache_file, key="data"))
                yield df
            elif self.config.output_format == "feather":
                yield pd.read_feather(cache_file)


class DistributedATLASProcessor:
    """
    Distributed processing for very large ATLAS datasets using Dask.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = None

    def setup_cluster(self, scheduler_address: Optional[str] = None):
        """Setup Dask distributed cluster."""
        if scheduler_address:
            self.client = Client(scheduler_address)
        else:
            self.client = Client(
                n_workers=self.config.n_workers,
                threads_per_worker=2,
                memory_limit=f"{self.config.max_memory_gb}GB",
            )

        logger.info(f"Dask cluster setup: {self.client}")

    @delayed
    def process_file_delayed(
        self, file_path: str, processor_func: Callable
    ) -> pd.DataFrame:
        """Process a single file using Dask delayed."""
        config = ProcessingConfig()  # Use default config for workers
        processor = LargeScaleATLASProcessor(config)

        branches = [
            "AnalysisJetsAuxDyn.pt",
            "AnalysisJetsAuxDyn.eta",
            "AnalysisJetsAuxDyn.phi",
            "AnalysisJetsAuxDyn.m",
            "AnalysisJetsAuxDyn.DFCommonJets_fJvt",
        ]

        chunks = []
        for chunk in processor.stream_atlas_files([file_path], branches):
            processed_chunk = processor_func(chunk)
            chunks.append(processed_chunk)

        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame()

    def process_dataset_distributed(
        self, file_paths: List[str], processor_func: Callable
    ) -> pd.DataFrame:
        """Process entire dataset using distributed computing."""
        if not self.client:
            self.setup_cluster()

        delayed_tasks = [
            self.process_file_delayed(file_path, processor_func)
            for file_path in file_paths
        ]

        logger.info(
            f"Processing {len(delayed_tasks)} files across {self.config.n_workers} workers"
        )
        results = []

        for future in tqdm(
            as_completed(delayed_tasks),
            total=len(delayed_tasks),
            desc="Processing Files",
        ):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")

        if results:
            final_result = pd.concat(results, ignore_index=True)
            logger.info(
                f"Combined {len(results)} results into {len(final_result)} events"
            )
            return final_result
        else:
            return pd.DataFrame()


class HDF5DatasetWriter:
    """HDF5 writer for large datasets."""

    def __init__(self, output_path: str, compression: str = "gzip"):
        self.output_path = output_path
        self.compression = compression
        self.file = None
        self.datasets = {}

    def __enter__(self):
        self.file = h5py.File(self.output_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def create_datasets(self, column_names: List[str], initial_size: int = 0):
        """Create resizable HDF5 datasets."""
        if self.file is None:
            raise RuntimeError("HDF5 file not opened. Use context manager.")

        for col in column_names:
            self.datasets[col] = self.file.create_dataset(
                col,
                shape=(initial_size,),
                maxshape=(None,),
                compression=self.compression,
                shuffle=True,
                fletcher32=True,
            )

    def append_chunk(self, chunk_df: pd.DataFrame):
        """Append a chunk to the HDF5 datasets."""
        if not self.datasets:
            self.create_datasets(chunk_df.columns.tolist())

        current_size = self.datasets[list(self.datasets.keys())[0]].shape[0]
        new_size = current_size + len(chunk_df)

        for col in chunk_df.columns:
            if col in self.datasets:
                self.datasets[col].resize((new_size,))
                self.datasets[col][current_size:new_size] = chunk_df[col].values


class ZarrDatasetWriter:
    """
    Zarr-based dataset writer for cloud-native storage.
    Excellent for distributed access and analysis.
    """

    def __init__(self, output_path: str, compression: str = "blosc"):
        self.output_path = output_path
        self.compression = compression
        self.store = None
        self.arrays = {}

    def __enter__(self):
        self.store = zarr.open(self.output_path, mode="w")
        self.root = self.store
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Zarr handles cleanup automatically

    def create_arrays(self, column_info: Dict[str, Tuple[str, int]]):
        """Create Zarr arrays with specified dtypes and chunk sizes."""
        for col, (dtype, chunk_size) in column_info.items():
            self.arrays[col] = self.root.create_dataset(
                col,
                shape=(0,),
                dtype=dtype,
                chunks=(chunk_size,),
                compression=self.compression,
            )

    def append_chunk(self, chunk_df: pd.DataFrame):
        """Append chunk to Zarr arrays."""
        for col in chunk_df.columns:
            if col in self.arrays:
                self.arrays[col].append(chunk_df[col].values)


if __name__ == "__main__":
    pass
