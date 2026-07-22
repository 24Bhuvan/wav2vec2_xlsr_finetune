from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class EmissionGenerator:
    """
    Loads the exported fine-tuned model and generates frame-level emissions.
    """

    def __init__(self, model_dir="../inference_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root = Path(model_dir)

        model_path = root / "model"
        processor_path = root / "processor"

        print(f"Initializing acoustic inference pipeline from: {root} on [{self.device}]")

        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

        # ==================================================
        # DIAGNOSTIC SECTION: TOKENIZER INFORMATION
        # ==================================================
        tokenizer = self.processor.tokenizer
        print("==================================================")
        print("TOKENIZER INFORMATION")
        print("==================================================")
        print(f"Pad Token       : {tokenizer.pad_token}")
        print(f"Pad Token ID    : {tokenizer.pad_token_id}")
        print(f"Vocabulary Size : {len(tokenizer)}")
        print()

        # Retrieve vocab, map to string tokens, and sort by token ID
        vocab = tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Ensure we print up to 35 IDs within valid vocabulary bounds
        max_idx = min(35, len(tokenizer))
        for i in range(max_idx):
            token = id_to_token.get(i, "[UNKNOWN_ID]")
            print(f"{i} -> {token}")
        print()
        # ==================================================

        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)

        self.model.eval()

    def generate_emissions(self, audio_path):
        waveform, sample_rate = sf.read(audio_path)

        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            )
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        input_values = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            emissions = torch.log_softmax(logits, dim=-1).squeeze(0).cpu()

        total_samples = waveform.shape[-1]
        total_frames = emissions.shape[0]

        frame_duration = (total_samples / 16000.0) / total_frames

        return emissions, frame_duration
