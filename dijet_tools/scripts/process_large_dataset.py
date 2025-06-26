#!/usr/bin/env python3
"""
Script for processing ATLAS datasets.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from dijet_tools.data.loaders import ATLASDataLoader
from dijet_tools.data.processors import (LargeScaleATLASProcessor,
                                         ProcessingConfig)
from dijet_tools.physics.kinematics import DijetKinematics
from dijet_tools.utils.config import ConfigManager
from dijet_tools.utils.logging import setup_logging


def main():
    """Entry point for dijet-process command."""
    parser = argparse.ArgumentParser(
        description="Process ATLAS datasets for dijet analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-files",
        "-i",
        required=True,
        help="Glob pattern for input ROOT files (e.g., 'data/*.root')",
    )

    parser.add_argument(
        "--output-dir", "-o", required=True, help="Output directory for processed data"
    )

    parser.add_argument(
        "--config", "-c", default="configs/default.yaml", help="Configuration file path"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of events per processing chunk",
    )

    parser.add_argument(
        "--max-memory-gb", type=float, default=64.0, help="Maximum memory usage in GB"
    )

    parser.add_argument(
        "--cache-dir", default="cache", help="Cache directory for intermediate results"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        config = ConfigManager.load_config(args.config)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_pattern = Path(args.input_files)
        input_files = list(input_pattern.parent.glob(input_pattern.name))

        if not input_files:
            logger.error(f"No files found matching pattern: {args.input_files}")
            sys.exit(1)

        logger.info(f"Found {len(input_files)} input files")

        processor_config = ProcessingConfig(
            chunk_size=args.chunk_size,
            max_memory_gb=args.max_memory_gb,
            cache_dir=args.cache_dir,
        )
        processor = LargeScaleATLASProcessor(processor_config)

        for i, file_path in enumerate(input_files):
            logger.info(f"Processing file {i+1}/{len(input_files)}: {file_path.name}")

            loader = ATLASDataLoader(str(file_path))

            def process_chunk(chunk_df):
                """Process chunk to extract dijet events."""
                processed_events = []

                for idx, event in chunk_df.iterrows():
                    if "AnalysisJetsAuxDyn.pt" in event:
                        pts = event["AnalysisJetsAuxDyn.pt"]
                        etas = event["AnalysisJetsAuxDyn.eta"]
                        phis = event["AnalysisJetsAuxDyn.phi"]
                        masses = event["AnalysisJetsAuxDyn.m"]

                        if hasattr(pts, "tolist"):
                            pts = pts.tolist()
                            etas = etas.tolist()
                            phis = phis.tolist()
                            masses = masses.tolist()

                        pt_threshold = 50.0
                        good_jets = [
                            (pt, eta, phi, m)
                            for pt, eta, phi, m in zip(pts, etas, phis, masses)
                            if pt > pt_threshold
                        ]

                        if len(good_jets) >= 2:
                            good_jets.sort(key=lambda x: x[0], reverse=True)

                            leading_pt, leading_eta, leading_phi, leading_m = good_jets[
                                0
                            ]
                            (
                                subleading_pt,
                                subleading_eta,
                                subleading_phi,
                                subleading_m,
                            ) = good_jets[1]

                            event_record = {
                                "event_index": idx,
                                "leading_jet_pt": leading_pt,
                                "leading_jet_eta": leading_eta,
                                "leading_jet_phi": leading_phi,
                                "leading_jet_m": leading_m,
                                "subleading_jet_pt": subleading_pt,
                                "subleading_jet_eta": subleading_eta,
                                "subleading_jet_phi": subleading_phi,
                                "subleading_jet_m": subleading_m,
                                "n_jets": len(good_jets),
                            }

                            kinematics = DijetKinematics()

                            cos_theta_star = kinematics.calculate_cos_theta_star(
                                leading_eta, subleading_eta
                            )
                            event_record["cos_theta_star"] = cos_theta_star

                            mjj = kinematics.calculate_mass(
                                leading_pt,
                                leading_eta,
                                leading_phi,
                                leading_m,
                                subleading_pt,
                                subleading_eta,
                                subleading_phi,
                                subleading_m,
                            )
                            delta_y = kinematics.calculate_rapidity_separation(
                                leading_eta, subleading_eta
                            )
                            delta_phi = kinematics.calculate_azimuthal_separation(
                                leading_phi, subleading_phi
                            )
                            chi = kinematics.calculate_chi_variable(
                                leading_eta, subleading_eta
                            )

                            event_record.update(
                                {
                                    "mjj": mjj,
                                    "delta_y": delta_y,
                                    "delta_phi": delta_phi,
                                    "chi": chi,
                                    "pt_balance": subleading_pt / leading_pt,
                                }
                            )

                            processed_events.append(event_record)

                return pd.DataFrame(processed_events)

            processed_chunks = []
            for chunk in processor.stream_atlas_files(
                [str(file_path)], loader.jet_branches
            ):
                processed_chunk = process_chunk(chunk)
                if len(processed_chunk) > 0:
                    processed_chunks.append(processed_chunk)

            if processed_chunks:
                processed_data = pd.concat(processed_chunks, ignore_index=True)

                output_file = output_dir / f"processed_{file_path.stem}.parquet"
                processed_data.to_parquet(output_file, index=False)

                logger.info(f"Saved processed data to: {output_file}")
            else:
                logger.warning(f"No valid events found in {file_path.name}")

        logger.info("Data processing complete!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
