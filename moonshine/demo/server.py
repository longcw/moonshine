"""WebSocket server for Moonshine STT transcription."""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from tokenizers import Tokenizer

# Local import of Moonshine ONNX model
MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel  # noqa: E402

SAMPLING_RATE = 16000


# Configure logging
class LogFormatter(logging.Formatter):
    def format(self, record):
        # Set default duration if not present
        if not hasattr(record, "duration"):
            record.duration = 0.0
        return super().format(record)


formatter = LogFormatter(
    fmt='%(asctime)s | %(levelname)s | dur=%(duration).3fs | text="%(message)s"',
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure the logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicate logs
for handler in logger.handlers[:-1]:
    logger.removeHandler(handler)


class TranscriptionServer:
    def __init__(
        self,
        model_name: str,
        host: str = "localhost",
        port: int = 8765,
        max_connections: Optional[int] = 100,
        debug_audio: bool = False,
        debug_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.active_connections = set()
        self.model = MoonshineOnnxModel(model_name=model_name)

        # Debug audio settings
        self.debug_audio = debug_audio
        if debug_audio:
            self.debug_dir = Path(debug_dir or "debug_audio")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug audio will be saved to {self.debug_dir}")

        # Initialize tokenizer
        tokenizer_path = os.path.join(
            MOONSHINE_DEMO_DIR, "..", "assets", "tokenizer.json"
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Warm up the model
        self.transcribe(np.zeros(SAMPLING_RATE, dtype=np.float32))

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk using Moonshine model."""
        # Normalize int16 to float32 range [-1, 1]
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        tokens = self.model.generate(audio_float[np.newaxis, :].astype(np.float32))
        return self.tokenizer.decode_batch(tokens)[0]

    def save_debug_audio(
        self, audio_data: np.ndarray, client_id: int, transcription: str
    ):
        """Save audio chunk to WAV file for debugging."""
        import wave

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Create a sanitized version of transcription for filename
        safe_trans = "".join(c if c.isalnum() else "_" for c in transcription)[:30]
        filename = f"{timestamp}_client{client_id}_{safe_trans}.wav"
        filepath = self.debug_dir / filename

        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLING_RATE)
            wav_file.writeframes(audio_data.tobytes())

        logger.info(f"Saved debug audio to {filename}", extra={"duration": 0})

    async def handle_connection(self, websocket):
        """Handle WebSocket connection and audio transcription."""
        # Check connection limit
        if (
            self.max_connections
            and len(self.active_connections) >= self.max_connections
        ):
            await websocket.close(1013, "Maximum connections reached")
            return

        client_id = id(websocket)
        self.active_connections.add(websocket)

        try:
            async for message in websocket:
                start_time = time.time()

                try:
                    # Convert bytes to int16 numpy array
                    audio_data = np.frombuffer(message, dtype=np.int16)

                    # Transcribe audio
                    transcription = self.transcribe(audio_data)
                    duration = time.time() - start_time

                    # Save debug audio if enabled
                    if self.debug_audio:
                        self.save_debug_audio(audio_data, client_id, transcription)

                    # Log the transcription and duration
                    logger.info(
                        transcription,
                        extra={
                            "duration": duration,
                            "client_id": client_id,
                            "audio_length": len(audio_data) / SAMPLING_RATE,
                        },
                    )

                    await websocket.send(transcription)

                except Exception as e:
                    logger.error(
                        f"Transcription error: {str(e)}", extra={"duration": 0}
                    )
                    await websocket.send(f"Error during transcription: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected", extra={"duration": 0})
        except Exception as e:
            logger.error(f"Connection error: {str(e)}", extra={"duration": 0})
        finally:
            self.active_connections.remove(websocket)

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=10,  # Wait 10 seconds for pong response
        ):
            logger.info(
                f"Moonshine STT Server running on ws://{self.host}:{self.port}",
                extra={"duration": 0},
            )
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="Moonshine STT WebSocket Server")
    parser.add_argument(
        "--model_name",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--max_connections",
        type=int,
        default=100,
        help="Maximum number of simultaneous connections",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Save audio chunks for debugging",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="debug_audio",
        help="Directory to save debug audio files (default: debug_audio)",
    )

    args = parser.parse_args()

    logger.info(
        f"Loading Moonshine model '{args.model_name}' (using ONNX runtime) ...",
        extra={"duration": 0},
    )

    server = TranscriptionServer(
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        max_connections=args.max_connections,
        debug_audio=args.debug_audio,
        debug_dir=args.debug_dir,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
