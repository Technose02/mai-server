# mai-server

A Rust-based server for generative AI inference with multiple model backends.

TODO:
- error-handling and logging
- apikey for security-layer AND llama.cpp
- reintegrate UI (clever dedicated routing needed)

## Overview

mai-server is a modular, high-performance server architecture designed for serving large language models and other generative AI systems. The project combines:
- A flexible gateway server (gw-server)
- Multiple inference backends (including llama.cpp)
- Process management utilities

The system is designed to support various model architectures and inference scenarios while maintaining a clean, maintainable codebase.

## Architecture

### Core Components

**1. gw-server**
- HTTP gateway for client requests
- Supports multiple model types and configurations
- Includes TLS integration (Let's Encrypt)
- Modular design with pluggable backends

**2. inference-backends**
- **llama.cpp backend**: Integration with llama.cpp for efficient LLM inference
  - Supports quantized models (Q8_0, etc.)
  - Handles model loading and execution
  - Configurable through JSON files
- **ComfyUI backend**: For image generation workflows

**3. managed-process**
- Process management utilities
- Handles external process execution and monitoring
- Supports both local and remote processes

## Key Features

- **Modular Design**: Easy to add new backends or modify existing ones
- **Performance**: Optimized for high-throughput inference
- **Flexibility**: Supports multiple model types and configurations
- **Security**: Built-in TLS with automatic certificate management
- **Extensibility**: Well-structured codebase for future development

## Getting Started

### Prerequisites
- Rust 1.70+
- Cargo
- Git
- Docker (optional, for some backend components)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mai-server.git

# Navigate to project root
cd mai-server

# Build the project
cargo build --release
```

### Configuration
The system uses JSON configuration files for model definitions. Key files include:
- `gw-server/http/*.json`: Model configurations
- `inference-backends/examples/cli_controlled.rs`: Example usage

## Usage

### Running the Server
```bash
# Start the server
cargo run --release --bin gw-server
```

### Connecting Clients
Clients can connect via HTTP to the gateway server, which routes requests to appropriate backends based on model configuration.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Submit a pull request

## License
MIT License - see LICENSE file for details

## Contact
For questions or support, please open an issue on GitHub.
