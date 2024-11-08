"""Example client for Moonshine STT WebSocket Server."""

import argparse
import asyncio
import time
import wave
from pathlib import Path

import numpy as np
import websockets


async def send_audio_file(
    websocket,
    audio_path: str,
    chunk_duration_ms: int = 1000,
    sample_rate: int = 16000,
):
    """Send audio file in chunks to the server and receive transcriptions."""
    chunk_size = int(sample_rate * (chunk_duration_ms / 1000))
    total_audio_duration = 0
    total_process_time = 0
    chunks_processed = 0
    
    with wave.open(audio_path, "rb") as wav_file:
        # Verify audio format
        if wav_file.getnchannels() != 1:
            raise ValueError("Audio must be mono")
        if wav_file.getframerate() != sample_rate:
            raise ValueError(f"Audio must have {sample_rate}Hz sampling rate")
        if wav_file.getsampwidth() != 2:  # 16-bit
            raise ValueError("Audio must be 16-bit")

        while True:
            frames = wav_file.readframes(chunk_size)
            if not frames:
                break
            
            chunk_start_time = time.time()
            
            # Send raw int16 audio bytes directly
            await websocket.send(frames)
            
            # Receive transcription
            response = await websocket.recv()
            
            # Calculate timing
            chunk_process_time = time.time() - chunk_start_time
            chunk_duration = len(frames) / (2 * sample_rate)  # 2 bytes per sample
            
            total_audio_duration += chunk_duration
            total_process_time += chunk_process_time
            chunks_processed += 1
            
            if response.startswith("Error:"):
                print(f"Server error: {response}")
            else:
                print(f"Chunk {chunks_processed}: {response}")
                print(f"  Processing time: {chunk_process_time:.3f}s")
                print(f"  Real-time factor: {chunk_duration/chunk_process_time:.2f}x")
    
    # Print final statistics
    print("\nProcessing Summary:")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total processing time: {total_process_time:.2f}s")
    print(f"Average real-time factor: {total_audio_duration/total_process_time:.2f}x")
    print(f"Chunks processed: {chunks_processed}")
    print(f"Average time per chunk: {total_process_time/chunks_processed:.3f}s")


async def main(audio_path: str, server_url: str, chunk_duration_ms: int):
    """Connect to STT server and process audio file."""
    try:
        start_time = time.time()
        async with websockets.connect(server_url) as websocket:
            print(f"Connected to {server_url}")
            print(f"Processing audio file: {audio_path}")
            print(f"Chunk duration: {chunk_duration_ms}ms\n")
            await send_audio_file(
                websocket, 
                audio_path, 
                chunk_duration_ms=chunk_duration_ms
            )
        total_time = time.time() - start_time
        print(f"Total wall time: {total_time:.2f}s")
    except websockets.exceptions.ConnectionError:
        print(f"Could not connect to server at {server_url}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moonshine STT Client Example")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to WAV audio file (16kHz, mono, 16-bit)",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=1000,
        help="Duration of audio chunks in milliseconds",
    )

    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        print(f"Audio file not found: {args.audio_path}")
        exit(1)

    asyncio.run(main(args.audio_path, args.server, args.chunk_duration)) 