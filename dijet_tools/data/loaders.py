import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import uproot
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class ATLASDataLoader:
    def __init__(
        self, file_paths: Union[str, List[str]], tree_name: str = "CollectionTree"
    ):
        """
        Initialize ATLAS data loader.

        Args:
            file_paths: Single file path or list of paths to ATLAS ROOT files
            tree_name: Name of tree in ROOT files (default: CollectionTree for PHYSLITE)
        """
        self.file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.tree_name = tree_name

        # Standard ATLAS jet branches
        self.jet_branches = [
            "AnalysisJetsAuxDyn.pt",
            "AnalysisJetsAuxDyn.eta",
            "AnalysisJetsAuxDyn.phi",
            "AnalysisJetsAuxDyn.m",
            "AnalysisJetsAuxDyn.DFCommonJets_fJvt",
        ]

        # Validate files exist
        self._validate_files()

    def _validate_files(self):
        """Validate that all input files exist and are accessible."""
        missing_files = []
        for file_path in self.file_paths:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")

        logger.info(f"Validated {len(self.file_paths)} ATLAS files")

    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        total_size = (
            sum(Path(f).stat().st_size for f in self.file_paths) / 1024**3
        )  # GB

        # Sample first file to get event count estimate
        total_events = 0
        root_file = uproot.open(self.file_paths[0])
        tree = root_file[self.tree_name]
        avg_events_per_file = tree.num_entries  # type: ignore
        total_events = avg_events_per_file * len(self.file_paths)

        return {
            "n_files": len(self.file_paths),
            "total_size_gb": total_size,
            "estimated_events": total_events,
            "avg_events_per_file": avg_events_per_file,
        }

    def list_available_branches(self, file_index: int = 0) -> List[str]:
        """List all available branches in the files."""
        root_file = uproot.open(self.file_paths[file_index])
        tree = root_file[self.tree_name]
        branches = list(tree.keys())
        return branches

    def stream_events(
        self,
        chunk_size: int = 50000,
        max_events: Optional[int] = None,
        branches: Optional[List[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream events from ATLAS files in chunks.

        Args:
            chunk_size: Number of events per chunk
            max_events: Maximum total events to read (None for all)
            branches: Specific branches to read (None for default jet branches)

        Yields:
            DataFrame chunks containing ATLAS events
        """
        if branches is None:
            branches = self.jet_branches

        events_read = 0

        with tqdm(
            total=len(self.file_paths), desc="Processing Files", unit="files"
        ) as pbar:
            for file_idx, file_path in enumerate(self.file_paths):
                if max_events and events_read >= max_events:
                    break

                try:
                    root_file = uproot.open(file_path)
                    tree = root_file[self.tree_name]
                    file_events = tree.num_entries  # type: ignore

                    # Process file in chunks
                    for start_idx in range(0, file_events, chunk_size):
                        if max_events and events_read >= max_events:
                            break

                        # Calculate chunk boundaries
                        end_idx = min(start_idx + chunk_size, file_events)
                        if max_events:
                            end_idx = min(
                                end_idx, start_idx + (max_events - events_read)
                            )

                        # Read chunk
                        chunk_data = tree.arrays(  # type: ignore
                            branches,
                            entry_start=start_idx,
                            entry_stop=end_idx,
                            library="pd",
                        )

                        # Add metadata
                        chunk_data["file_index"] = file_idx
                        chunk_data["event_start_index"] = start_idx

                        events_read += len(chunk_data)
                        yield chunk_data

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue

                pbar.update(1)
                pbar.set_postfix({"events": f"{events_read:,}"})
