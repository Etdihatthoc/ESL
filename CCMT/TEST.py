"""
Example usage of the CCMT models for ESL grading

This script demonstrates how to:
1. Create the CCMT model
2. Create the dataset 
3. Perform forward pass
4. Save/load models
"""

import torch
import pandas as pd
import os
from torch.utils.data import DataLoader

# Import from models package
from models import (
    create_esl_ccmt_model,
    create_esl_ccmt_dataset,
    get_ccmt_collate_fn,
    TextProcessor,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_DATASET_CONFIG
)


def example_model_creation():
    """Example of creating CCMT model"""
    print("=== Creating CCMT Model ===")
    
    # Option 1: Use default configuration
    model = create_esl_ccmt_model()
    print(f"Model created with default config")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Option 2: Custom configuration
    custom_config = {
        "common_dim": 256,  # Smaller dimension
        "ccmt_depth": 4,    # Fewer layers
        "dropout": 0.1      # Less dropout
    }
    model_small = create_esl_ccmt_model(config=custom_config)
    print(f"Small model created with {sum(p.numel() for p in model_small.parameters()):,} parameters")
    
    return model


def example_dataset_creation():
    """Example of creating CCMT dataset"""
    print("\n=== Creating CCMT Dataset ===")
    
    # Create sample dataframe (replace with your actual data)
    # Note: Using more realistic data that won't be filtered out
    sample_data = {
        'text': [
            '"Hello, how are you today? I am fine and ready to learn English."', 
            '"I like to study English very much. It helps me communicate with people from different countries."',
            '"My name is John and I come from Vietnam. I have been studying English for three years."'
        ],
        'grammar': [7.5, 6.0, 8.0],
        'question_type': [1, 2, 3],
        'absolute_path': [
            '/path/to/audio1.wav', 
            '/path/to/audio2.wav', 
            '/path/to/audio3.wav'
        ]
    }
    df = pd.DataFrame(sample_data)
    
    print(f"Original dataframe: {len(df)} samples")
    print("Sample data:")
    for i, row in df.iterrows():
        print(f"  {i+1}. Text: {row['text'][:50]}...")
        print(f"     Grammar: {row['grammar']}, Type: {row['question_type']}")
    
    # Initialize text processor (might take some time to download models)
    print("\nInitializing text processor (downloading models if needed)...")
    try:
        text_processor = TextProcessor(
            asr_model_name="openai/whisper-base",
            translation_model_name="Helsinki-NLP/opus-mt-en-vi"
        )
        print("Text processor initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize text processor: {e}")
        print("Using None - will skip text processing in dataset")
        text_processor = None
    
    # Create dataset with less strict filtering for the example
    try:
        dataset = create_esl_ccmt_dataset(
            dataframe=df,
            text_processor=text_processor,
            is_train=True,
            remove_low_content=False,  # Disable for example
            filter_scores=False,       # Disable for example
            use_text_cache=True
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty after filtering!")
            print("This might be due to:")
            print("- Audio files don't exist (this is expected in the example)")
            print("- Text processing failed")
            print("- Filtering criteria too strict")
            return None, None
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Creating a minimal dataset for demonstration...")
        
        # Create a simpler dataset configuration
        dataset = create_esl_ccmt_dataset(
            dataframe=df,
            text_processor=None,  # Skip text processor
            is_train=False,       # No augmentation
            remove_low_content=False,
            filter_scores=False,
            use_text_cache=False
        )
        
        print(f"Minimal dataset created with {len(dataset)} samples")
    
    # Only create dataloader if dataset is not empty
    if len(dataset) > 0:
        # Create dataloader
        collate_fn = get_ccmt_collate_fn()
        dataloader = DataLoader(
            dataset,
            batch_size=min(2, len(dataset)),  # Ensure batch_size <= dataset size
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for debugging
        )
        
        print(f"DataLoader created with batch size {min(2, len(dataset))}")
        
        # Test getting one sample
        try:
            print("\nTesting dataset sample:")
            sample = dataset[0]
            print("Sample keys:", list(sample.keys()))
            print(f"Audio shape: {sample['audio_chunks'].shape}")
            print(f"English text length: {len(sample['english_text'])}")
            print(f"Score: {sample['score'].item()}")
            
            # Test dataloader
            print("\nTesting dataloader:")
            batch = next(iter(dataloader))
            print("Batch keys:", list(batch.keys()))
            print(f"Batch audio shape: {batch['audio_chunks'].shape}")
            print(f"Batch scores: {batch['score']}")
            
        except Exception as e:
            print(f"Warning: Could not test dataset/dataloader: {e}")
        
        return dataset, dataloader
    else:
        print("Cannot create dataloader with empty dataset")
        return None, None


def example_forward_pass():
    """Example of forward pass through the model"""
    print("\n=== Forward Pass Example ===")
    
    # Create model
    model = create_esl_ccmt_model()
    model.eval()
    
    # Create dummy batch data
    batch_size = 2
    num_tokens = DEFAULT_MODEL_CONFIG["num_tokens_per_modality"]
    seq_len = DEFAULT_DATASET_CONFIG["max_text_length"]
    num_chunks = DEFAULT_DATASET_CONFIG["num_audio_chunks"]
    chunk_samples = DEFAULT_DATASET_CONFIG["chunk_length_sec"] * DEFAULT_DATASET_CONFIG["sample_rate"]
    
    dummy_batch = {
        'audio_chunks': torch.randn(batch_size, num_chunks, chunk_samples),
        'english_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'english_attention_mask': torch.ones(batch_size, seq_len),
        'vietnamese_input_ids': torch.randint(0, 1000, (batch_size, seq_len)), 
        'vietnamese_attention_mask': torch.ones(batch_size, seq_len),
    }
    
    print(f"Input shapes:")
    for key, value in dummy_batch.items():
        print(f"  {key}: {value.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            audio_chunks=dummy_batch['audio_chunks'],
            english_input_ids=dummy_batch['english_input_ids'],
            english_attention_mask=dummy_batch['english_attention_mask'],
            vietnamese_input_ids=dummy_batch['vietnamese_input_ids'],
            vietnamese_attention_mask=dummy_batch['vietnamese_attention_mask']
        )
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nExpected scores: {outputs['expected_score']}")
    
    return outputs


def example_save_load():
    """Example of saving and loading model"""
    print("\n=== Save/Load Example ===")
    
    # Create and save model
    model = create_esl_ccmt_model()
    save_path = "checkpoints/example_ccmt_model.pth"
    
    model.save(save_path)
    
    # Load model
    loaded_model = model.load(save_path)
    
    # Verify they produce same output
    dummy_input = {
        'audio_chunks': torch.randn(1, 10, 16000*30),
        'english_input_ids': torch.randint(0, 1000, (1, 256)),
        'english_attention_mask': torch.ones(1, 256),
        'vietnamese_input_ids': torch.randint(0, 1000, (1, 256)),
        'vietnamese_attention_mask': torch.ones(1, 256),
    }
    
    with torch.no_grad():
        original_output = model(**dummy_input)
        loaded_output = loaded_model(**dummy_input)
    
    difference = torch.abs(original_output['expected_score'] - loaded_output['expected_score']).max()
    print(f"Max difference between original and loaded model: {difference.item():.6f}")
    
    if difference < 1e-6:
        print("✓ Save/load successful!")
    else:
        print("✗ Save/load failed!")
    
    # Clean up
    try:
        os.remove(save_path)
        os.rmdir("checkpoints")
        print("Cleaned up temporary files")
    except:
        pass


def main():
    """Run all examples"""
    print("CCMT Model Examples")
    print("=" * 50)
    
    try:
        # Example 1: Model creation
        model = example_model_creation()
        
        # Example 2: Dataset creation  
        dataset, dataloader = example_dataset_creation()
        
        # Example 3: Forward pass
        outputs = example_forward_pass()
        
        # Example 4: Save/load
        example_save_load()
        
        print("\n" + "=" * 50)
        if dataset is not None:
            print("All examples completed successfully!")
        else:
            print("Examples completed with some warnings (dataset creation issues)")
            print("This is expected when running without actual audio files")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()