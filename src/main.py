"""CLI entry-point orchestrating C-SADA smoke test / full experiment."""

import argparse
import logging
import os
import sys
import yaml
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_smoke_test() -> None:
    """Run smoke test configuration for quick validation."""
    print("=== C-SADA Smoke Test ===")
    config_path = "config/smoke_test.yaml"
    
    try:
        config = load_config(config_path)
        print(f"Loaded configuration: {config_path}")
        print(f"Experiment: {config['experiment']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Datasets: {config['datasets']}")
        print(f"Models: {config['models']}")
        
        output_dir = config.get("output_dir", ".research/iteration2")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        from .train import run_experiment
        results = run_experiment(config_path)
        
        from .evaluate import create_visualization
        plot_path = create_visualization(results, output_dir)
        print(f"Visualization saved to: {plot_path}")
        
        print("\n=== Smoke Test Completed Successfully ===")
        
    except Exception as e:
        import traceback
        print(f"Smoke test failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


def run_full_experiment() -> None:
    """Run full-scale experiment configuration."""
    print("=== C-SADA Full Experiment ===")
    config_path = "config/full_experiment.yaml"
    
    try:
        config = load_config(config_path)
        print(f"Loaded configuration: {config_path}")
        print(f"Experiment: {config['experiment']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Datasets: {config['datasets']}")
        print(f"Models: {config['models']}")
        
        output_dir = config.get("output_dir", ".research/iteration2")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        from .train import run_experiment
        results = run_experiment(config_path)
        
        from .evaluate import create_visualization
        plot_path = create_visualization(results, output_dir)
        print(f"Visualization saved to: {plot_path}")
        
        print("\n=== Full Experiment Completed Successfully ===")
        
    except Exception as e:
        print(f"Full experiment failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for C-SADA experiments."""
    parser = argparse.ArgumentParser(
        description="CERTIFIED-SADA (C-SADA) Research Experiment Framework"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smoke-test", 
        action="store_true", 
        help="Run smoke-test config for quick validation"
    )
    group.add_argument(
        "--full-experiment", 
        action="store_true", 
        help="Run full-scale experiment"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.smoke_test:
        run_smoke_test()
    elif args.full_experiment:
        run_full_experiment()


if __name__ == "__main__":
    main()
