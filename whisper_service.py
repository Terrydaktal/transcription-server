"""Compatibility entry point for the local transcription server."""

import sys

from assistant_transcription import server

if __name__ == "__main__":
    sys.exit(server.main())

sys.modules[__name__] = server
