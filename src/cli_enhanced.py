#!/usr/bin/env python3
"""
Enhanced CLI for Sign Language Detector
Provides a comprehensive command-line interface for all project operations
"""

import argparse
import sys
import os
from pathlib import Path
from src.core import collect_images, create_dataset, train_classifier, inference_classifier
from src.io_ import ask, print_prompt


class SignLanguageCLI:
    """Enhanced CLI handler for Sign Language Detector"""
    
    def __init__(self):
        self.parser = self._build_parser()
    
    def _build_parser(self):
        """Build the argument parser with all commands"""
        parser = argparse.ArgumentParser(
            description="Sign Language Detector - ASL Recognition System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive mode
  python -m src.cli_enhanced
  
  # Collect images for 5 classes, 100 images each
  python -m src.cli_enhanced collect --classes 5 --images 100 --output my_data
  
  # Create dataset from collected images
  python -m src.cli_enhanced dataset --classes 5 --name my_dataset
  
  # Train Random Forest model
  python -m src.cli_enhanced train --dataset my_dataset --model forest
  
  # Train Neural Network model
  python -m src.cli_enhanced train --dataset my_dataset --model neural --classes 5
  
  # Run inference with trained model
  python -m src.cli_enhanced infer --model forest
  
  # List available datasets and models
  python -m src.cli_enhanced list
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Interactive mode (default)
        parser.add_argument('--interactive', '-i', action='store_true',
                          help='Run in interactive mode (default if no command specified)')
        
        # Collect images command
        collect_parser = subparsers.add_parser('collect', help='Collect training images from webcam')
        collect_parser.add_argument('--classes', '-c', type=int, required=True,
                                   help='Number of classes (letters) to collect')
        collect_parser.add_argument('--images', '-n', type=int, required=True,
                                   help='Number of images per class')
        collect_parser.add_argument('--output', '-o', type=str, required=True,
                                   help='Output folder name for collected images')
        
        # Create dataset command
        dataset_parser = subparsers.add_parser('dataset', help='Create dataset from collected images')
        dataset_parser.add_argument('--classes', '-c', type=int, required=True,
                                   help='Number of classes in the dataset')
        dataset_parser.add_argument('--name', '-n', type=str, required=True,
                                   help='Name for the dataset file (without extension)')
        
        # Train model command
        train_parser = subparsers.add_parser('train', help='Train a classification model')
        train_parser.add_argument('--dataset', '-d', type=str, required=True,
                                 help='Dataset file name (without extension)')
        train_parser.add_argument('--model', '-m', type=str, required=True,
                                 choices=['forest', 'neural', 'f', 'n'],
                                 help='Model type: forest/f (Random Forest) or neural/n (Neural Network)')
        train_parser.add_argument('--classes', '-c', type=int,
                                 help='Number of classes (required for neural network)')
        
        # Inference command
        infer_parser = subparsers.add_parser('infer', help='Run inference with trained model')
        infer_parser.add_argument('--model', '-m', type=str, required=True,
                                 choices=['forest', 'neural', 'f', 'n'],
                                 help='Model type: forest/f or neural/n')
        
        # List command
        subparsers.add_parser('list', help='List available datasets and models')
        
        # Setup command
        subparsers.add_parser('setup', help='Setup project directories and download models')
        
        # Info command
        subparsers.add_parser('info', help='Show project information and status')
        
        return parser
    
    def run(self, args=None):
        """Execute the CLI with given arguments"""
        parsed_args = self.parser.parse_args(args)
        
        # If no command specified or interactive flag, run interactive mode
        if not parsed_args.command or parsed_args.interactive:
            self._interactive_mode()
            return
        
        # Execute the specified command
        command_map = {
            'collect': self._cmd_collect,
            'dataset': self._cmd_dataset,
            'train': self._cmd_train,
            'infer': self._cmd_infer,
            'list': self._cmd_list,
            'setup': self._cmd_setup,
            'info': self._cmd_info,
        }
        
        handler = command_map.get(parsed_args.command)
        if handler:
            try:
                handler(parsed_args)
            except KeyboardInterrupt:
                print_prompt("\n\nOperation cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print_prompt(f"\nError: {str(e)}")
                sys.exit(1)
        else:
            self.parser.print_help()
    
    def _interactive_mode(self):
        """Run the interactive menu-driven interface"""
        print_prompt("\n" + "="*60)
        print_prompt("  Sign Language Detector - Interactive Mode")
        print_prompt("="*60)
        print_prompt("\nWhat would you like to do?\n")
        
        self._ask_user_action()
    
    def _ask_user_action(self):
        """Ask user for action in interactive mode"""
        response = ask(
            "1 - Collect images\n"
            "2 - Create dataset\n"
            "3 - Train model\n"
            "4 - Use model (inference)\n"
            "5 - List datasets and models\n"
            "6 - Show project info\n"
            "7 - Quit\n"
            "\nSelect option: ",
            cast_type=int, min=1, max=7
        )
        
        action_finished = False
        
        if response == "1":
            action_finished = self._interactive_collect()
        elif response == "2":
            action_finished = self._interactive_dataset()
        elif response == "3":
            action_finished = self._interactive_train()
        elif response == "4":
            action_finished = self._interactive_infer()
        elif response == "5":
            self._cmd_list(None)
            action_finished = True
        elif response == "6":
            self._cmd_info(None)
            action_finished = True
        elif response == "7":
            print_prompt("\nGoodbye!")
            return
        
        if action_finished:
            print_prompt("\n" + "-"*60)
            print_prompt("Action completed successfully!")
            print_prompt("-"*60 + "\n")
            self._ask_user_action()
    
    def _interactive_collect(self):
        """Interactive image collection"""
        print_prompt("\n--- Image Collection ---")
        imgs_per_class = ask("How many images per class? ", cast_type=int, min=1, max=1000)
        num_classes = ask("How many classes? ", cast_type=int, min=1, max=26)
        folder_name = ask("Output folder name: ", cast_type=str)
        
        print_prompt(f"\nCollecting {imgs_per_class} images for {num_classes} classes...")
        return collect_images(int(num_classes), int(imgs_per_class), folder_name)
    
    def _interactive_dataset(self):
        """Interactive dataset creation"""
        print_prompt("\n--- Dataset Creation ---")
        num_classes = ask("Number of classes: ", cast_type=int, min=1, max=26)
        dataset_name = ask("Dataset name (without extension): ", cast_type=str)
        
        print_prompt(f"\nCreating dataset '{dataset_name}' with {num_classes} classes...")
        return create_dataset(int(num_classes), dataset_name)
    
    def _interactive_train(self):
        """Interactive model training"""
        print_prompt("\n--- Model Training ---")
        
        # Check for available datasets
        dataset_dir = Path('./src/data_pickle')
        if dataset_dir.exists():
            datasets = list(dataset_dir.glob('*.pickle'))
            if datasets:
                print_prompt("\nAvailable datasets:")
                for i, ds in enumerate(datasets, 1):
                    print_prompt(f"  {i}. {ds.stem}")
        
        data_file = ask("\nDataset name (without extension): ", cast_type=str)
        
        # Check if dataset exists
        if not (dataset_dir / f'{data_file}.pickle').exists():
            print_prompt(f"\nError: Dataset '{data_file}.pickle' not found!")
            return False
        
        # Ask for model type
        model_type = None
        while model_type not in ['f', 'n']:
            model_type = ask("\nModel type (f=Random Forest, n=Neural Network): ", cast_type=str).lower()
            if model_type not in ['f', 'n']:
                print_prompt("Invalid choice. Please enter 'f' or 'n'.")
        
        num_classes = None
        if model_type == 'n':
            num_classes = ask("Number of classes: ", cast_type=int, min=1, max=26)
        
        print_prompt(f"\nTraining {'Random Forest' if model_type == 'f' else 'Neural Network'} model...")
        return train_classifier(data_file, model_type, int(num_classes) if num_classes else None)
    
    def _interactive_infer(self):
        """Interactive inference"""
        print_prompt("\n--- Model Inference ---")
        
        # Check for available models
        model_dir = Path('./src/models')
        if model_dir.exists():
            models = list(model_dir.glob('model_*.p'))
            if models:
                print_prompt("\nAvailable models:")
                for model in models:
                    model_type = "Random Forest" if "_f.p" in model.name else "Neural Network"
                    print_prompt(f"  - {model.name} ({model_type})")
        
        model_type = None
        while model_type not in ['f', 'n']:
            model_type = ask("\nModel type (f=Random Forest, n=Neural Network): ", cast_type=str).lower()
            if model_type not in ['f', 'n']:
                print_prompt("Invalid choice. Please enter 'f' or 'n'.")
        
        print_prompt("\nStarting inference... Press 'Q' to begin, 'Q' again to finish.")
        return inference_classifier(model_type)
    
    def _cmd_collect(self, args):
        """Handle collect command"""
        print_prompt(f"\nCollecting {args.images} images for {args.classes} classes...")
        success = collect_images(args.classes, args.images, args.output)
        if success:
            print_prompt(f"\nImages saved to: ./src/data/{args.output}/")
        else:
            print_prompt("\nImage collection failed!")
            sys.exit(1)
    
    def _cmd_dataset(self, args):
        """Handle dataset command"""
        print_prompt(f"\nCreating dataset '{args.name}' with {args.classes} classes...")
        success = create_dataset(args.classes, args.name)
        if success:
            print_prompt(f"\nDataset saved to: ./src/data_pickle/{args.name}.pickle")
        else:
            print_prompt("\nDataset creation failed!")
            sys.exit(1)
    
    def _cmd_train(self, args):
        """Handle train command"""
        # Normalize model type
        model_type = 'f' if args.model in ['forest', 'f'] else 'n'
        
        # Check if dataset exists
        dataset_path = Path(f'./src/data_pickle/{args.dataset}.pickle')
        if not dataset_path.exists():
            print_prompt(f"\nError: Dataset '{args.dataset}.pickle' not found!")
            sys.exit(1)
        
        # Validate classes for neural network
        if model_type == 'n' and not args.classes:
            print_prompt("\nError: --classes is required for neural network training!")
            sys.exit(1)
        
        model_name = "Random Forest" if model_type == 'f' else "Neural Network"
        print_prompt(f"\nTraining {model_name} model on dataset '{args.dataset}'...")
        
        success = train_classifier(args.dataset, model_type, args.classes)
        if success:
            print_prompt(f"\nModel saved to: ./src/models/model_{model_type}.p")
        else:
            print_prompt("\nModel training failed!")
            sys.exit(1)
    
    def _cmd_infer(self, args):
        """Handle infer command"""
        model_type = 'f' if args.model in ['forest', 'f'] else 'n'
        
        # Check if model exists
        model_path = Path(f'./src/models/model_{model_type}.p')
        if not model_path.exists():
            print_prompt(f"\nError: Model 'model_{model_type}.p' not found!")
            print_prompt("Please train a model first.")
            sys.exit(1)
        
        model_name = "Random Forest" if model_type == 'f' else "Neural Network"
        print_prompt(f"\nStarting inference with {model_name} model...")
        print_prompt("Press 'Q' to begin, 'Q' again to finish.\n")
        
        success = inference_classifier(model_type)
        if not success:
            sys.exit(1)
    
    def _cmd_list(self, args):
        """Handle list command"""
        print_prompt("\n" + "="*60)
        print_prompt("  Available Resources")
        print_prompt("="*60)
        
        # List datasets
        print_prompt("\nDatasets:")
        dataset_dir = Path('./src/data_pickle')
        if dataset_dir.exists():
            datasets = sorted(dataset_dir.glob('*.pickle'))
            if datasets:
                for ds in datasets:
                    size = ds.stat().st_size / 1024  # KB
                    print_prompt(f"  - {ds.name} ({size:.1f} KB)")
            else:
                print_prompt("  (none found)")
        else:
            print_prompt("  (directory not found)")
        
        # List models
        print_prompt("\nTrained Models:")
        model_dir = Path('./src/models')
        if model_dir.exists():
            models = sorted(model_dir.glob('model_*.p'))
            if models:
                for model in models:
                    model_type = "Random Forest" if "_f.p" in model.name else "Neural Network"
                    size = model.stat().st_size / 1024  # KB
                    print_prompt(f"  - {model.name} ({model_type}, {size:.1f} KB)")
            else:
                print_prompt("  (none found)")
        else:
            print_prompt("  (directory not found)")
        
        # List data folders
        print_prompt("\nData Folders:")
        data_dir = Path('./src/data')
        if data_dir.exists():
            folders = [f for f in data_dir.iterdir() if f.is_dir()]
            if folders:
                for folder in sorted(folders):
                    classes = [c for c in folder.iterdir() if c.is_dir()]
                    print_prompt(f"  - {folder.name}/ ({len(classes)} classes)")
            else:
                print_prompt("  (none found)")
        else:
            print_prompt("  (directory not found)")
        
        print_prompt("")
    
    def _cmd_setup(self, args):
        """Handle setup command"""
        print_prompt("\n" + "="*60)
        print_prompt("  Project Setup")
        print_prompt("="*60)
        
        # Create necessary directories
        dirs = [
            './src/data',
            './src/data_pickle',
            './src/models'
        ]
        
        print_prompt("\nCreating project directories...")
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print_prompt(f"  ✓ {dir_path}")
        
        print_prompt("\nSetup complete!")
        print_prompt("\nNext steps:")
        print_prompt("  1. Collect training images: python -m src.cli_enhanced collect")
        print_prompt("  2. Create dataset: python -m src.cli_enhanced dataset")
        print_prompt("  3. Train model: python -m src.cli_enhanced train")
        print_prompt("  4. Run inference: python -m src.cli_enhanced infer")
        print_prompt("")
    
    def _cmd_info(self, args):
        """Handle info command"""
        print_prompt("\n" + "="*60)
        print_prompt("  Sign Language Detector - Project Information")
        print_prompt("="*60)
        
        print_prompt("\nDescription:")
        print_prompt("  ASL (American Sign Language) alphabet recognition system")
        print_prompt("  using computer vision and machine learning.")
        
        print_prompt("\nFeatures:")
        print_prompt("  - Webcam-based image collection")
        print_prompt("  - MediaPipe hand landmark detection")
        print_prompt("  - Random Forest classifier")
        print_prompt("  - Neural Network classifier (PyTorch)")
        print_prompt("  - Real-time inference with sentence correction")
        
        print_prompt("\nProject Structure:")
        print_prompt("  src/data/          - Collected training images")
        print_prompt("  src/data_pickle/   - Processed datasets")
        print_prompt("  src/models/        - Trained models")
        
        print_prompt("\nSupported Models:")
        print_prompt("  - Random Forest (sklearn)")
        print_prompt("  - Neural Network (PyTorch)")
        
        print_prompt("\nFor help: python -m src.cli_enhanced --help")
        print_prompt("")


def main():
    """Main entry point"""
    cli = SignLanguageCLI()
    cli.run()


if __name__ == '__main__':
    main()
