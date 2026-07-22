import os
import json
import logging
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional
from transformers import Wav2Vec2Processor
from processor_builder import build_or_load_processor

# ==============================================================================
# CONFIGURATION - HARDCODED CONSTANTS
# ==============================================================================
BATCH_SIZE = 8
DEFAULT_PROCESSOR_PATH = "pretrained/wav2vec2-xls-r-300m"
TRAIN_METADATA_PATH = "data/train/transcripts.csv"
VALID_METADATA_PATH = "data/valid/transcripts.csv"
TEST_METADATA_PATH = "data/test/transcripts.csv"
TARGET_SAMPLE_RATE = 16000

# Setup structured logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "dataset.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset")


class Wav2Vec2Dataset(Dataset):
    """
    Production-grade PyTorch Dataset for Wav2Vec2-XLS-R fine-tuning.
    Loads audio, transcripts, and word-level annotations. Tokenizes text 
    and preprocesses waveforms using Hugging Face's Wav2Vec2Processor.
    """
    def __init__(
        self,
        metadata_csv_path: str,
        processor_path_or_name: str = DEFAULT_PROCESSOR_PATH,
        project_root: str = "",
        processor: Optional[Wav2Vec2Processor] = None
    ) -> None:
        """
        Args:
            metadata_csv_path: Path to the transcripts.csv file for the split.
            processor_path_or_name: Local path or HF hub model name of the processor.
            project_root: Absolute path to the repository root. Defaults to current directory.
            processor: Optional pre-instantiated processor.
        """
        self.metadata_csv_path = metadata_csv_path
        self.project_root = project_root if project_root else os.getcwd()
        
        # Resolve CSV path relative to project root
        full_csv_path = os.path.join(self.project_root, metadata_csv_path)
        
        if not os.path.exists(full_csv_path):
            raise FileNotFoundError(f"Metadata file not found: {full_csv_path}")
            
        logger.info(f"Loading metadata from {full_csv_path}...")
        self.df = pd.read_csv(full_csv_path)
        self.metadata = self.df.to_dict(orient="records")
        logger.info(f"Successfully loaded {len(self.metadata)} records.")

        # Initialize processor
        if processor is not None:
            self.processor = processor
        else:
            processor_path_or_name = processor_path_or_name or DEFAULT_PROCESSOR_PATH
            # Build vocabulary from training transcripts on first run, or reload
            # the saved processor on subsequent runs. No Hugging Face downloads.
            self.processor = build_or_load_processor(
                processor_dir=processor_path_or_name,
                train_csv_path=TRAIN_METADATA_PATH,
                project_root=self.project_root,
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Loads, verifies, tokenizes, and returns a single sample.
        If a sample is corrupted or missing, it will recursively try the next sample.
        """
        attempts = 0
        max_attempts = len(self.metadata)
        
        while attempts < max_attempts:
            current_index = (index + attempts) % len(self.metadata)
            sample_info = self.metadata[current_index]
            attempts += 1
            
            try:
                # 1. Extract and verify paths from metadata
                audio_path = sample_info.get("audio_path")
                transcript = sample_info.get("transcript")
                annotation_path = sample_info.get("annotation_path")
                
                if not audio_path or not annotation_path:
                    raise ValueError(f"Missing audio_path or annotation_path in metadata for index {current_index}")
                
                # Resolve paths relative to project root
                full_audio_path = os.path.join(self.project_root, audio_path)
                full_annotation_path = os.path.join(self.project_root, annotation_path)
                
                # 2. Check existence of files
                if not os.path.exists(full_audio_path):
                    raise FileNotFoundError(f"Audio file not found: {full_audio_path}")
                if not os.path.exists(full_annotation_path):
                    raise FileNotFoundError(f"Annotation file not found: {full_annotation_path}")
                
                # 3. Handle empty transcripts
                if not isinstance(transcript, str) or len(transcript.strip()) == 0:
                    raise ValueError(f"Empty transcript for index {current_index}")
                
                # 4. Load audio with SoundFile (numpy -> torch)
                audio_np, sr = sf.read(full_audio_path, dtype="float32")
                # Convert to torch tensor. SoundFile returns shape (samples,) for mono or (samples, channels) for multi-channel.
                waveform = torch.from_numpy(audio_np)
                # Ensure shape (1, samples) and mono audio
                if waveform.dim() == 2:
                    # Average across channels -> mono, keep dimension for consistency
                    waveform = torch.mean(waveform, dim=1, keepdim=True)
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                # At this point waveform shape is (1, samples) and dtype is float32

                
                # Check sample rate (resample dynamically if mismatch occurs)
                if sr != TARGET_SAMPLE_RATE:
                    logger.warning(
                        f"Sample rate mismatch: {full_audio_path} is {sr}Hz. "
                        f"Resampling to {TARGET_SAMPLE_RATE}Hz."
                    )
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                    waveform = resampler(waveform)
                
                # Convert 2D mono (1, samples) to 1D (samples,)
                waveform_1d = waveform.squeeze(0)
                
                # 5. Load TXT word-level annotation
                with open(full_annotation_path, "r", encoding="utf-8") as f:
                    annotation_lines = [line.strip() for line in f if line.strip()]

                annotation = []
                for line in annotation_lines:
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 3:
                        start, end, word = parts[0], parts[1], parts[2]
                        annotation.append({
                            "start": float(start),
                            "end": float(end),
                            "word": word
                        })
                    else:
                        raise ValueError(f"Invalid annotation line format in {full_annotation_path}: {line}")
                
                # 6. Process audio waveform using HF Processor
                # Wav2Vec2Processor expects a 1D tensor or numpy array
                inputs = self.processor(waveform_1d, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
                input_values = inputs.input_values.squeeze(0)
                
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.squeeze(0)
                else:
                    # By default, Wav2Vec2 models do not strictly require attention masks for audio inputs
                    # unless specified, but we construct it if needed.
                    attention_mask = torch.ones_like(input_values, dtype=torch.long)

                # 7. Tokenize transcript to target labels (input_ids)
                labels = self.processor(text=transcript).input_ids
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                return {
                    "input_values": input_values,
                    "attention_mask": attention_mask,
                    "labels": labels_tensor,
                    "transcript": transcript,
                    "annotation": annotation,
                    "audio_path": audio_path,
                    "speaker_id": sample_info.get("speaker_id", ""),
                    "duration": float(sample_info.get("duration", 0.0))
                }
                
            except Exception as e:
                logger.error(
                    f"Error processing sample at index {current_index} (attempt {attempts}/{max_attempts}): {e}. "
                    f"Skipping and moving to next record..."
                )
                continue
                
        raise RuntimeError("All samples in the dataset have been exhausted or are corrupted.")


class Wav2Vec2Collate:
    """
    Custom collator to dynamically pad variable-length audio inputs 
    and target label sequences. Minimizes padding length per batch to 
    optimize GPU memory usage and training speed.
    """
    def __init__(self, processor: Wav2Vec2Processor) -> None:
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pads waveform sequences and text targets in the batch.
        
        Args:
            batch: A list of dicts returned by Wav2Vec2Dataset.__getitem__.
            
        Returns:
            A batch dictionary containing padded PyTorch tensors and metadata lists:
                - input_values: Padded float tensor of shape (batch_size, max_audio_len)
                - attention_mask: Padded long tensor of shape (batch_size, max_audio_len)
                - labels: Padded long tensor of shape (batch_size, max_label_len), padded with -100
                - transcripts: List of original transcripts
                - annotations: List of word-level annotation dicts
                - audio_paths: List of relative audio paths
                - speaker_ids: List of speaker IDs
                - durations: List of audio durations (float)
        """
        # Separate tensors and metadata
        input_values_list = [s["input_values"] for s in batch]
        attention_mask_list = [s["attention_mask"] for s in batch]
        labels_list = [s["labels"] for s in batch]
        
        # Dynamic padding for inputs (features/waveforms) using pad_sequence
        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values_list,
            batch_first=True,
            padding_value=0.0
        )
        
        # Dynamic padding for attention masks
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )
        
        # Dynamic padding for text label tokens. 
        # Hugging Face CTC Loss expects a padding value of -100 to ignore pad positions in loss calculation.
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )
        
        # Extract metadata fields
        transcripts = [s["transcript"] for s in batch]
        annotations = [s["annotation"] for s in batch]
        audio_paths = [s["audio_path"] for s in batch]
        speaker_ids = [s["speaker_id"] for s in batch]
        durations = [s["duration"] for s in batch]
        
        return {
            "input_values": input_values_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "transcripts": transcripts,
            "annotations": annotations,
            "audio_paths": audio_paths,
            "speaker_ids": speaker_ids,
            "durations": durations
        }


# ==============================================================================
# DATALOADER CREATION HELPERS
# ==============================================================================
def get_dataloader(
    csv_path: str,
    processor: Optional[Wav2Vec2Processor] = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    project_root: str = ""
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader configured with the custom Collate Function.
    """
    dataset = Wav2Vec2Dataset(
        metadata_csv_path=csv_path,
        processor_path_or_name=DEFAULT_PROCESSOR_PATH,
        project_root=project_root,
        processor=processor
    )
    
    # Initialize the custom collate function with the dataset's processor
    collate_fn = Wav2Vec2Collate(processor=dataset.processor)
    
    # Enable persistent workers only if multi-worker process is enabled
    persistent_workers = num_workers > 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    )


def get_train_dataloader(
    processor: Optional[Wav2Vec2Processor] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
    project_root: str = ""
) -> DataLoader:
    """
    Returns the DataLoader for the training dataset split.
    """
    return get_dataloader(
        csv_path=TRAIN_METADATA_PATH,
        processor=processor,
        batch_size=batch_size,
        shuffle=True,  # Shuffle enabled for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        project_root=project_root
    )


def get_valid_dataloader(
    processor: Optional[Wav2Vec2Processor] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
    project_root: str = ""
) -> DataLoader:
    """
    Returns the DataLoader for the validation dataset split.
    """
    return get_dataloader(
        csv_path=VALID_METADATA_PATH,
        processor=processor,
        batch_size=batch_size,
        shuffle=False,  # Shuffle disabled for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        project_root=project_root
    )


def get_test_dataloader(
    processor: Optional[Wav2Vec2Processor] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
    project_root: str = ""
) -> DataLoader:
    """
    Returns the DataLoader for the test dataset split.
    """
    return get_dataloader(
        csv_path=TEST_METADATA_PATH,
        processor=processor,
        batch_size=batch_size,
        shuffle=False,  # Shuffle disabled for testing
        num_workers=num_workers,
        pin_memory=pin_memory,
        project_root=project_root
    )


# ==============================================================================
# SELF-VERIFICATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RUNNING DATASET AND DATALOADER VERIFICATION TESTS")
    logger.info("=" * 60)
    
    # 1. Verify existence of CSV metadata
    splits = {
        "Train": TRAIN_METADATA_PATH,
        "Validation": VALID_METADATA_PATH,
        "Test": TEST_METADATA_PATH
    }
    
    for split_name, path in splits.items():
        if os.path.exists(path):
            logger.info(f"[SUCCESS] Found {split_name} metadata CSV: {path}")
        else:
            logger.error(f"[ERROR] Missing {split_name} metadata CSV: {path}")
            
    # 2. Try initializing Dataset & loading a sample
    try:
        # Load dataset
        logger.info("Initializing train dataset and loading processor...")
        train_dataset = Wav2Vec2Dataset(metadata_csv_path=TRAIN_METADATA_PATH)
        num_samples = len(train_dataset)
        logger.info(f"Number of samples in training split: {num_samples}")
        
        # Inspect single sample
        logger.info("Inspecting single dataset sample (index 0)...")
        sample = train_dataset[0]
        
        logger.info("-" * 40)
        logger.info(f"Keys in returned sample: {list(sample.keys())}")
        logger.info(f"Audio Path             : {sample['audio_path']}")
        logger.info(f"Transcript             : '{sample['transcript']}'")
        logger.info(f"Speaker ID             : {sample['speaker_id']}")
        logger.info(f"Duration               : {sample['duration']} seconds")
        logger.info(f"Input Values shape     : {sample['input_values'].shape} ({sample['input_values'].dtype})")
        logger.info(f"Attention Mask shape   : {sample['attention_mask'].shape} ({sample['attention_mask'].dtype})")
        logger.info(f"Labels shape           : {sample['labels'].shape} ({sample['labels'].dtype})")
        logger.info(f"Annotation preview     : {str(sample['annotation'])[:150]}...")
        logger.info("-" * 40)
        
        # 3. Test DataLoader & Custom Collate Function
        logger.info("Testing DataLoaders with custom collator...")
        train_loader = get_train_dataloader(processor=train_dataset.processor, num_workers=2)
        valid_loader = get_valid_dataloader(processor=train_dataset.processor, num_workers=2)
        test_loader = get_test_dataloader(processor=train_dataset.processor, num_workers=2)
        
        logger.info(f"Train Dataloader batches      : {len(train_loader)}")
        logger.info(f"Validation Dataloader batches : {len(valid_loader)}")
        logger.info(f"Test Dataloader batches       : {len(test_loader)}")
        
        # Print a single batch from training loader
        logger.info("Fetching and printing one training batch...")
        batch = next(iter(train_loader))
        
        logger.info("=" * 60)
        logger.info("BATCH SHAPE AND ALIGNMENT INSPECTION:")
        logger.info(f"Batch 'input_values' shape   : {batch['input_values'].shape}")
        logger.info(f"Batch 'attention_mask' shape : {batch['attention_mask'].shape}")
        logger.info(f"Batch 'labels' shape         : {batch['labels'].shape}")
        logger.info(f"Padded positions check (count -100 in labels): {torch.sum(batch['labels'] == -100).item()}")
        logger.info(f"Batch size (transcripts)     : {len(batch['transcripts'])}")
        logger.info(f"First transcript in batch    : '{batch['transcripts'][0]}'")
        logger.info("=" * 60)
        
        logger.info("[VERIFICATION COMPLETED SUCCESSFULLY] Dataset and DataLoader pipelines are fully functional.")
        
    except Exception as e:
        logger.exception(f"Verification failed: {e}")
