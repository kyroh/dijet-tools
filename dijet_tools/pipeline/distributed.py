import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dask import delayed
from dask.distributed import Client, Future, as_completed

from ..data.processors import LargeScaleATLASProcessor
from ..utils.config import AnalysisConfig

logger = logging.getLogger(__name__)


class DistributedPipeline:
    """
    Distributed computing pipeline for processing large ATLAS datasets.
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize distributed pipeline.

        Args:
            config: Complete analysis configuration
        """
        self.config = config
        self.client = None
        self.futures = []

    def setup_cluster(self, scheduler_address: Optional[str] = None):
        """
        Set up Dask distributed cluster.

        Args:
            scheduler_address: Address of Dask scheduler (None for local cluster)
        """
        if scheduler_address or self.config.compute.scheduler_address:
            address = scheduler_address or self.config.compute.scheduler_address
            self.client = Client(address)
            logger.info(f"Connected to Dask cluster at {address}")
        else:
            self.client = Client(
                n_workers=self.config.compute.n_workers,
                threads_per_worker=2,
                memory_limit=self.config.compute.memory_limit_per_worker,
            )
            logger.info(f"Created local Dask cluster: {self.client}")

        logger.info(f"Cluster dashboard: {self.client.dashboard_link}")

    @delayed
    def process_file_chunk(
        self, file_paths: List[str], chunk_processor: Callable
    ) -> pd.DataFrame:
        """
        Process a chunk of files using Dask delayed.

        Args:
            file_paths: List of ATLAS files to process
            chunk_processor: Function to process the files

        Returns:
            Processed DataFrame
        """
        try:
            processor = LargeScaleATLASProcessor(self.config.data)

            result = chunk_processor(file_paths, processor)

            logger.info(
                f"Worker processed {len(file_paths)} files, "
                f"result shape: {result.shape if hasattr(result, 'shape') else len(result)}"
            )

            return result

        except Exception as e:
            logger.error(f"Worker failed processing {len(file_paths)} files: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    def distribute_file_processing(
        self, all_files: List[str], chunk_processor: Callable, files_per_chunk: int = 5
    ) -> List[Future]:
        """
        Distribute file processing across workers.

        Args:
            all_files: List of all ATLAS files to process
            chunk_processor: Function to process file chunks
            files_per_chunk: Number of files per processing chunk

        Returns:
            List of Dask futures
        """
        if not self.client:
            self.setup_cluster()

        logger.info(f"Distributing {len(all_files)} files across workers")
        logger.info(f"Files per chunk: {files_per_chunk}")

        file_chunks = [
            all_files[i : i + files_per_chunk]
            for i in range(0, len(all_files), files_per_chunk)
        ]

        logger.info(f"Created {len(file_chunks)} processing chunks")

        futures = []
        for i, chunk in enumerate(file_chunks):
            future = self.client.submit(
                self.process_file_chunk, chunk, chunk_processor, key=f"chunk-{i:04d}"
            )
            futures.append(future)

        self.futures.extend(futures)
        logger.info(f"Submitted {len(futures)} tasks to cluster")

        return futures

    def collect_results(
        self, futures: List[Future], progress_callback: Optional[Callable] = None
    ) -> List[pd.DataFrame]:
        """
        Collect results from distributed processing.

        Args:
            futures: List of Dask futures to collect
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed DataFrames
        """
        logger.info(f"Collecting results from {len(futures)} tasks...")

        results = []
        completed_count = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)

                completed_count += 1

                if progress_callback:
                    progress_callback(completed_count, len(futures))

                if completed_count % 10 == 0:
                    logger.info(f"Completed {completed_count}/{len(futures)} tasks")

            except Exception as e:
                logger.error(f"Task failed: {e}")
                completed_count += 1

        logger.info(
            f"Collected {len(results)} successful results from {len(futures)} tasks"
        )
        return results

    def run_distributed_analysis(
        self, file_paths: List[str], output_path: str, chunk_processor: Callable
    ) -> Dict[str, Any]:
        """
        Run complete distributed analysis pipeline.

        Args:
            file_paths: List of ATLAS files to process
            output_path: Path to save final results
            chunk_processor: Function to process file chunks

        Returns:
            Analysis results and metadata
        """
        start_time = time.time()

        if not self.client:
            self.setup_cluster()

        futures = self.distribute_file_processing(
            file_paths,
            chunk_processor,
            files_per_chunk=self.config.data.chunk_size
            // 10000,
        )

        def progress_callback(completed, total):
            percent = (completed / total) * 100
            logger.info(f"Progress: {completed}/{total} ({percent:.1f}%)")

        chunk_results = self.collect_results(futures, progress_callback)

        if chunk_results:
            logger.info("Combining chunk results...")
            final_result = pd.concat(chunk_results, ignore_index=True)

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_path.endswith(".parquet"):
                final_result.to_parquet(output_path, compression="snappy")
            elif output_path.endswith(".h5"):
                final_result.to_hdf(output_path, key="data", mode="w")
            else:
                final_result.to_pickle(output_path)

            logger.info(f"Final results saved to {output_path}")
        else:
            raise RuntimeError("No successful processing results obtained")

        total_time = time.time() - start_time
        metadata = {
            "total_files": len(file_paths),
            "total_events": len(final_result),
            "processing_time": total_time,
            "events_per_second": len(final_result) / total_time,
            "n_workers": len(self.client.scheduler_info()["workers"]),
            "chunks_processed": len(chunk_results),
            "output_file": str(output_path),
        }

        logger.info(f"Distributed analysis complete:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Events processed: {len(final_result):,}")
        logger.info(f"  Processing rate: {metadata['events_per_second']:.1f} events/s")

        return {"data": final_result, "metadata": metadata}

    def shutdown_cluster(self):
        """Shutdown the Dask cluster."""
        if self.client:
            self.client.close()
            logger.info("Dask cluster shut down")


def example_chunk_processor(
    file_paths: List[str], processor: LargeScaleATLASProcessor
) -> pd.DataFrame:
    """
    Example chunk processor for distributed processing.

    Args:
        file_paths: Files to process in this chunk
        processor: Processor instance

    Returns:
        Processed DataFrame
    """
    from ..data.loaders import ATLASDataLoader
    from ..data.selectors import ATLASEventSelector
    from ..physics.kinematics import DijetKinematics

    loader = ATLASDataLoader(file_paths)
    selector = ATLASEventSelector()
    kinematics = DijetKinematics()

    all_events = []

    for chunk in loader.stream_events(chunk_size=10000):
        selected = selector.apply_jet_quality_cuts(chunk)

        if not selected.empty:
            physics_data = kinematics.calculate_all_observables(selected)
            all_events.append(physics_data)

    if all_events:
        return pd.concat(all_events, ignore_index=True)
    else:
        return pd.DataFrame()
