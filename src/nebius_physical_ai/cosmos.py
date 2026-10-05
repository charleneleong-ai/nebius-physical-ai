"""What the Cosmos server is called and which image and model it runs; shared by the client and the A100 launcher."""

SERVED_MODEL_NAME = (
    "cosmos"  # what clients put in `model`; keep in sync with COSMOS_SERVED_NAME in scripts/cosmos_local.sh
)
COSMOS3_IMAGE = "vllm/vllm-openai:cosmos3"  # vLLM >= 0.23 with native Cosmos 3 Reasoner support (NVIDIA's cookbook)
COSMOS3_NANO = "nvidia/Cosmos3-Nano"  # not gated (OpenMDW licence), so no Hugging Face token is needed
